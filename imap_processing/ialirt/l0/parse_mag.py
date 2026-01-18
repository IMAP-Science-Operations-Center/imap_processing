"""Functions to support I-ALiRT MAG packet parsing."""

import logging
from decimal import Decimal

import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline

from imap_processing.ialirt.l0.ialirt_spice import (
    transform_instrument_vectors_to_inertial,
)
from imap_processing.ialirt.l0.mag_l0_ialirt_data import (
    Packet0,
    Packet1,
    Packet2,
    Packet3,
)
from imap_processing.ialirt.utils.grouping import (
    _populate_instrument_header_items,
    find_groups,
)
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.mag.l1a.mag_l1a_data import TimeTuple
from imap_processing.mag.l1b.mag_l1b import (
    calibrate_vector,
    shift_time,
)
from imap_processing.mag.l1d.mag_l1d_data import MagL1d
from imap_processing.mag.l2.mag_l2_data import MagL2L1dBase, ValidFrames
from imap_processing.spice.geometry import (
    SpiceFrame,
    cartesian_to_spherical,
    frame_transform,
    spherical_to_cartesian,
)
from imap_processing.spice.time import met_to_ttj2000ns, ttj2000ns_to_et

logger = logging.getLogger(__name__)


def get_pkt_counter(status_values: xr.DataArray) -> xr.DataArray:
    """
    Get the packet counters.

    Parameters
    ----------
    status_values : xr.DataArray
        Status data.

    Returns
    -------
    pkt_counters : xr.DataArray
        Packet counters.
    """
    # mag_status is a 24 bit unsigned field
    # The leading 2 bits of STATUS are a 2 bit 0-3 counter
    pkt_counter = (status_values >> 22) & 0x03

    return pkt_counter


def get_status_data(status_values: xr.DataArray, pkt_counters: xr.DataArray) -> dict:
    """
    Get the status data.

    Parameters
    ----------
    status_values : xr.DataArray
        Status data.
    pkt_counters : xr.DataArray
        Packet counters.

    Returns
    -------
    combined_packets : dict
        Decoded packets.
    """
    decoders = {
        0: Packet0,
        1: Packet1,
        2: Packet2,
        3: Packet3,
    }

    combined_packets = {}

    for pkt_num, decoder in decoders.items():
        status_subset = status_values[pkt_counters == pkt_num]
        decoded_packet = decoder(int(status_subset))
        combined_packets.update(vars(decoded_packet))

    return combined_packets


def get_bytes(val: int) -> list[int]:
    """
    Extract three bytes from a 24-bit integer.

    Parameters
    ----------
    val : int
        24-bit integer value.

    Returns
    -------
    list[int]
        List of three extracted bytes.
    """
    return [
        (val >> 16) & 0xFF,  # Most significant byte (Byte2)
        (val >> 8) & 0xFF,  # Middle byte (Byte1)
        (val >> 0) & 0xFF,  # Least significant byte (Byte0)
    ]


def extract_magnetic_vectors(science_values: xr.DataArray) -> dict:
    """
    Extract the magnetic vectors.

    Parameters
    ----------
    science_values : xr.DataArray
        Science data.

    Returns
    -------
    vectors : dict
        Magnetic vectors.
    """
    # Primary sensor:
    pri_x: np.int16 = np.uint16((int(science_values[0]) >> 8) & 0xFFFF).astype(np.int16)
    pri_y: np.int16 = np.uint16(
        ((int(science_values[0]) << 8) & 0xFF00)
        | ((int(science_values[1]) >> 16) & 0xFF)
    ).astype(np.int16)
    pri_z: np.int16 = np.uint16(int(science_values[1]) & 0xFFFF).astype(np.int16)

    # Secondary sensor:
    sec_x: np.int16 = np.uint16((int(science_values[2]) >> 8) & 0xFFFF).astype(np.int16)
    sec_y: np.int16 = np.uint16(
        ((int(science_values[2]) << 8) & 0xFF00)
        | ((int(science_values[3]) >> 16) & 0xFF)
    ).astype(np.int16)

    sec_z: np.int16 = np.uint16(int(science_values[3]) & 0xFFFF).astype(np.int16)

    vectors = {
        "pri_x": pri_x,
        "pri_y": pri_y,
        "pri_z": pri_z,
        "sec_x": sec_x,
        "sec_y": sec_y,
        "sec_z": sec_z,
    }

    return vectors


def get_time(
    grouped_data: xr.Dataset,
    group: int,
    pkt_counter: xr.DataArray,
    time_shift_mago: xr.DataArray,
    time_shift_magi: xr.DataArray,
) -> dict:
    """
    Get the time for the grouped data.

    Parameters
    ----------
    grouped_data : xr.Dataset
        Grouped data.
    group : int
        Group number.
    pkt_counter : xr.DataArray
        Packet counter.
    time_shift_mago : xr.DataArray
        Time shift value mago.
    time_shift_magi : xr.DataArray
        Time shift value magi.

    Returns
    -------
    time_data : dict
        Coarse and fine time for Primary and Secondary Sensors.

    Notes
    -----
    Packet id 0 is course and fine time for the primary sensor PRI.
    Packet id 2 is the course time for the secondary sensor SEC.
    """
    # Get the coarse and fine time for the primary and secondary sensors.
    pri_coarsetm = grouped_data["mag_acq_tm_coarse"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 0]

    pri_fintm = grouped_data["mag_acq_tm_fine"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 0]

    sec_coarsetm = grouped_data["mag_acq_tm_coarse"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 2]

    sec_fintm = grouped_data["mag_acq_tm_fine"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 2]

    time_data: dict[str, int | float] = {
        "pri_coarsetm": int(pri_coarsetm.item()),
        "pri_fintm": int(pri_fintm.item()),
        "sec_coarsetm": int(sec_coarsetm.item()),
        "sec_fintm": int(sec_fintm.item()),
    }

    primary_time = TimeTuple(int(pri_coarsetm.item()), int(pri_fintm.item()))
    secondary_time = TimeTuple(int(sec_coarsetm.item()), int(sec_fintm.item()))

    time_data_primary_ttj2000ns = primary_time.to_j2000ns()
    time_data["primary_epoch"] = shift_time(
        time_data_primary_ttj2000ns, time_shift_mago
    )
    time_data_secondary_ttj2000ns = secondary_time.to_j2000ns()
    time_data["secondary_epoch"] = shift_time(
        time_data_secondary_ttj2000ns, time_shift_magi
    )

    return time_data


def calculate_l1b(
    grouped_data: xr.Dataset,
    group: int,
    pkt_counter: xr.DataArray,
    science_data: dict,
    status_data: dict,
    calibration_dataset: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Calculate equivalent of l1b data product.

    Parameters
    ----------
    grouped_data : xr.Dataset
        Grouped data.
    group : int
        Group number.
    pkt_counter : xr.DataArray
        Packet counter.
    science_data : dict
        Science data.
    status_data : dict
        Status data.
    calibration_dataset : xr.Dataset
        Calibration dataset.

    Returns
    -------
    updated_vector_mago : numpy.ndarray
        Calibrated mago vector.
    updated_vector_magi : numpy.ndarray
        Calibrated magi vector.
    time_data : dict
        Time data.
    """
    calibration_matrix_mago, time_shift_mago = (
        retrieve_matrix_from_single_l1b_calibration(calibration_dataset, is_mago=True)
    )
    calibration_matrix_magi, time_shift_magi = (
        retrieve_matrix_from_single_l1b_calibration(calibration_dataset, is_mago=False)
    )

    logger.info(f"calibration_matrix_mago shape: {calibration_matrix_mago.shape}.")
    logger.info(f"calibration_matrix_magi shape: {calibration_matrix_magi.shape}.")

    # Get time values for each group.
    time_data = get_time(
        grouped_data, group, pkt_counter, time_shift_mago, time_shift_magi
    )

    input_vector_mago = np.array(
        [
            science_data["pri_x"],
            science_data["pri_y"],
            science_data["pri_z"],
            status_data["fob_range"],
        ]
    )
    input_vector_magi = np.array(
        [
            science_data["sec_x"],
            science_data["sec_y"],
            science_data["sec_z"],
            status_data["fib_range"],
        ]
    )

    updated_vector_mago = calibrate_vector(input_vector_mago, calibration_matrix_mago)
    updated_vector_magi = calibrate_vector(input_vector_magi, calibration_matrix_magi)

    return updated_vector_mago, updated_vector_magi, time_data


def calibrate_and_offset_vectors(
    vectors: np.ndarray,
    calibration: np.ndarray,
    offsets: np.ndarray,
    is_magi: bool = False,
) -> np.ndarray:
    """
    Apply calibration and offsets to magnetic vectors.

    Parameters
    ----------
    vectors : np.ndarray
        Raw magnetic vectors, shape (n, 4).
    calibration : np.ndarray
        Calibration matrix, shape (3, 3, 4).
    offsets : np.ndarray
        Offsets array, shape (2, 4, 3) where:
        - index 0 = MAGo, 1 = MAGi
        - second index = range (0–3)
        - third index = axis (x, y, z)
    is_magi : bool, optional
        True if applying to MAGi data, False for MAGo.

    Returns
    -------
    calibrated_and_offset_vectors : np.ndarray
        Calibrated and offset vectors, shape (n, 3).
    """
    # Apply calibration matrix -> (n,4)
    # apply_calibration_offset_single_vector
    calibrated = MagL2L1dBase.apply_calibration(vectors.reshape(1, 4), calibration)

    # Apply offsets per vector
    # vec shape (4)
    # offsets shape (2, 4, 3) where first index is 0 for MAGo and 1 for MAGi
    calibrated = np.array(
        [
            MagL1d.apply_calibration_offset_single_vector(vec, offsets, is_magi=is_magi)
            for vec in calibrated
        ]
    )

    return calibrated[:, :3]


def apply_gradiometry_correction(
    mago_vectors_eclipj2000: np.ndarray,
    mago_time_data: np.ndarray,
    magi_vectors_eclipj2000: np.ndarray,
    magi_time_data: np.ndarray,
    gradiometer_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align MAGi to MAGo timestamps and apply gradiometry correction.

    Parameters
    ----------
    mago_vectors_eclipj2000 : np.ndarray
        MAGo vectors in inertial frame, shape (N, 3).
    mago_time_data : np.ndarray
        Time for primary sensor, shape (N, 3).
    magi_vectors_eclipj2000 : np.ndarray
        MAGi vectors in inertial frame, shape (M, 3).
    magi_time_data : np.ndarray
        Time for secondary sensor, shape (N, 3).
    gradiometer_factor : np.ndarray
        A (3,3) element matrix to scale and rotate the gradiometer offsets.

    Returns
    -------
    mago_corrected : np.ndarray
        Corrected MAGo vectors in inertial frame, shape (N, 3).
    magnitude : np.ndarray
        Magnitude of corrected MAGo vectors, shape (N,).
    """
    gradiometry_offsets = MagL1d.calculate_gradiometry_offsets(
        mago_vectors_eclipj2000,
        mago_time_data,
        magi_vectors_eclipj2000,
        magi_time_data,
    )
    mago_corrected = MagL1d.apply_gradiometry_offsets(
        gradiometry_offsets, mago_vectors_eclipj2000, gradiometer_factor
    )
    magnitude = np.linalg.norm(mago_corrected, axis=-1).squeeze()

    return mago_corrected, magnitude


def interpolate_spherical(
    sc_inertial_right: np.ndarray,
    sc_inertial_decline: np.ndarray,
    sc_spin_phase: np.ndarray,
    attitude_time: np.ndarray,
    target_time: float,
) -> tuple:
    """
    Interpolate spherical coordinates.

    Parameters
    ----------
    sc_inertial_right : numpy.ndarray
        Inertial right ascension for 4 packets 0 to 360 degrees, shape (4).
    sc_inertial_decline : numpy.ndarray
        Inertial declination for 4 packets -45 to 45 degrees, shape (4).
    sc_spin_phase : numpy.ndarray
        Spin phase for 4 packets 0 to 360 degrees, shape (4).
    attitude_time : np.ndarray
        Timestamps for all packets in ttj2000ns.
    target_time : float
        Time at which to apply the transformation.
        Will be primary_epoch (mago vector) or secondary_epoch (magi vector).
        Example: time_data['primary_epoch'].

    Returns
    -------
    ra_deg np.ndarray
        Interpolated right ascension based on time (deg).
    dec_deg np.ndarray
        Interpolated declination based on time (deg).
    spin_phase_deg np.ndarray
        Interpolated spin-phase based on time (deg).
    """
    # Interpolate spin phase, RA, and Dec at target_time
    # Convert RA/Dec to unit cartesian vectors
    spherical_coords = np.stack(
        [
            np.ones_like(sc_inertial_right),
            sc_inertial_right,
            sc_inertial_decline,
        ],
        axis=-1,
    )
    vecs = spherical_to_cartesian(spherical_coords)

    # This was chosen instead of linear interpolation
    # to account for the vector moving along a curved
    # arc on the unit sphere.
    spline_x = CubicSpline(attitude_time, vecs[:, 0])
    spline_y = CubicSpline(attitude_time, vecs[:, 1])
    spline_z = CubicSpline(attitude_time, vecs[:, 2])

    # Interpolate in Cartesian space
    vx = float(spline_x(target_time))
    vy = float(spline_y(target_time))
    vz = float(spline_z(target_time))

    v_interp = np.array([vx, vy, vz])
    # Normalize vector so that its magnitude is 1.
    v_interp /= np.linalg.norm(v_interp)

    # Convert back to spherical
    ra_dec = cartesian_to_spherical(v_interp)
    ra_deg = ra_dec[1]
    dec_deg = ra_dec[2]

    # Account for discontinuities in spin phase.
    spin_phase_unwrapped = np.unwrap(np.radians(sc_spin_phase))
    spin_phase_interp = np.interp(target_time, attitude_time, spin_phase_unwrapped)
    spin_phase_deg = np.degrees(spin_phase_interp) % 360

    return ra_deg, dec_deg, spin_phase_deg


def transform_to_inertial(
    sc_spin_phase_rad: np.ndarray,
    sc_inertial_right: np.ndarray,
    sc_inertial_decline: np.ndarray,
    attitude_time: np.ndarray,
    target_time: float,
    mag_vector: np.ndarray,
    instrument_frame: SpiceFrame,
) -> np.ndarray:
    """
    Transform vector to ECLIPJ2000.

    Parameters
    ----------
    sc_spin_phase_rad : numpy.ndarray
        Spin phase for 4 packets 0 to 2π radians, shape (4).
    sc_inertial_right : numpy.ndarray
        Inertial right ascension for 4 packets 0 to 2π radians, shape (4).
    sc_inertial_decline : numpy.ndarray
        Inertial declination for 4 packets -π/2 to π/2 radians, shape (4).
    attitude_time : np.ndarray
        Timestamps for all packets in ttj2000ns.
    target_time : float
        Time at which to apply the transformation.
        Will be primary_epoch (mago vector) or secondary_epoch (magi vector).
        Example: time_data['primary_epoch'].
    mag_vector : numpy.ndarray
        Vector, shape (3).
    instrument_frame : SpiceFrame
        SPICE frame of the instrument.

    Returns
    -------
    inertial_vector : np.ndarray
        Transformed vector in the ECLIPJ2000 frame, shape (3,).

    Notes
    -----
    The MAG vectors are calculated based on 4 packets,
    each of which contains its own spin phase,
    inertial right ascension, and inertial decline.
    """
    if target_time < attitude_time.min() or target_time > attitude_time.max():
        logger.warning(
            f"target_time {target_time} is outside attitude_time bounds "
            f"[{attitude_time.min()}, {attitude_time.max()}]; using edge values."
        )

    # Get sort order based on attitude_time
    sort_idx = np.argsort(attitude_time)

    # Sort all arrays accordingly
    attitude_time = attitude_time[sort_idx]
    sc_spin_phase_rad = sc_spin_phase_rad[sort_idx]
    sc_inertial_right = sc_inertial_right[sort_idx]
    sc_inertial_decline = sc_inertial_decline[sort_idx]

    ra_deg, dec_deg, spin_phase_deg = interpolate_spherical(
        np.degrees(sc_inertial_right),
        np.degrees(sc_inertial_decline),
        np.degrees(sc_spin_phase_rad),
        attitude_time,
        target_time,
    )

    # Transform each into ECLIPJ2000
    inertial_vector = transform_instrument_vectors_to_inertial(
        np.asarray(mag_vector).reshape(1, 3),
        np.array([spin_phase_deg]),
        np.array([ra_deg]),
        np.array([dec_deg]),
        instrument_frame,
    )[0]

    return inertial_vector


def transform_to_frames(
    target_time: np.ndarray,
    inertial_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform vector to different frames.

    Parameters
    ----------
    target_time : np.ndarray
        Time at which to apply the transformation.
        Will be primary_epoch (mago vector).
        Example: time_data['primary_epoch'].
    inertial_vector : np.ndarray
        Transformed vector in the ECLIPJ2000 frame, shape (3,).

    Returns
    -------
    gse_vector : np.ndarray
        Transformed vector in the GSE frame, shape (3,).
    gsm_vector : np.ndarray
        Transformed vector in the GSM frame, shape (3,).
    rtn_vector : np.ndarray
        Transformed vector in the RTN frame, shape (3,).
    """
    et_target_time = ttj2000ns_to_et(target_time)

    gse_vector = frame_transform(
        et_target_time, inertial_vector, SpiceFrame.ECLIPJ2000, SpiceFrame.IMAP_GSE
    )
    gsm_vector = frame_transform(
        et_target_time, inertial_vector, SpiceFrame.ECLIPJ2000, SpiceFrame.IMAP_GSM
    )
    rtn_vector = frame_transform(
        et_target_time, inertial_vector, SpiceFrame.ECLIPJ2000, SpiceFrame.IMAP_RTN
    )

    return gse_vector, gsm_vector, rtn_vector


def process_packet(
    accumulated_data: xr.Dataset,
    engineering_calibration_dataset: xr.Dataset,
    l1d_calibration_dataset: xr.Dataset,
) -> list[dict]:
    """
    Parse the MAG packets.

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.
    engineering_calibration_dataset : xr.Dataset
        Engineering calibration dataset.
    l1d_calibration_dataset : xr.Dataset
        L1D calibration dataset.

    Returns
    -------
    mag_data : list[dict]
        Dictionaries of the parsed data product.
    """
    logger.info(
        f"Parsing MAG for time: {accumulated_data['mag_acq_tm_coarse'].min().values} - "
        f"{accumulated_data['mag_acq_tm_coarse'].max().values}."
    )

    # Subsecond time conversion specified in 7516-9054 GSW-FSW ICD.
    # Value of SCLK subseconds, unsigned, (LSB = 1/256 sec)
    met = calculate_time(
        accumulated_data["sc_sclk_sec"], accumulated_data["sc_sclk_sub_sec"], 256
    )

    # Add required parameters.
    accumulated_data["met"] = met
    pkt_counter = get_pkt_counter(accumulated_data["mag_status"])
    accumulated_data["pkt_counter"] = pkt_counter

    # Convert from incrementing uint16 (0-65535) to radians.
    sc_spin_phase_rad = accumulated_data["sc_spin_phase"].astype(float) * (
        2 * np.pi / 65535.0
    )
    sc_inertial_right = accumulated_data["sc_inertial_right"].astype(float) * (
        0.0055 * np.pi / 180
    )
    sc_inertial_decline = accumulated_data["sc_inertial_decline"].astype(float) * (
        0.0027 * np.pi / 180
    )

    attitude_time = met_to_ttj2000ns(accumulated_data["met"])

    grouped_data = find_groups(accumulated_data, (0, 3), "pkt_counter", "met")

    unique_groups = np.unique(grouped_data["group"])
    mag_data = []
    met_all = []
    mago_vectors_all = []
    mago_times_all = []
    magi_vectors_all = []
    magi_times_all = []
    incomplete_groups = []

    for group in unique_groups:
        # Get status values for each group.
        status_values = grouped_data["mag_status"][
            (grouped_data["group"] == group).values
        ]
        pkt_counter = grouped_data["pkt_counter"][
            (grouped_data["group"] == group).values
        ]

        if not np.array_equal(pkt_counter, np.arange(4)):
            incomplete_groups.append(group)
            continue

        # Get decoded status data.
        status_data = get_status_data(status_values, pkt_counter)

        if status_data["pri_isvalid"] == 0 and status_data["sec_isvalid"] == 0:
            logger.info(f"Group {group} contains no valid data for either sensor.")
            continue

        # Get science values for each group.
        science_values = grouped_data["mag_data"][
            (grouped_data["group"] == group).values
        ]
        science_data = extract_magnetic_vectors(science_values)
        updated_vector_mago, updated_vector_magi, time_data = calculate_l1b(
            grouped_data,
            group,
            pkt_counter,
            science_data,
            status_data,
            engineering_calibration_dataset,
        )

        # Note: primary = MAGo, secondary = MAGi.
        # Populate with a FILL value if either sensor is invalid,
        # but not both.
        if status_data["pri_isvalid"] == 0:
            updated_vector_mago = np.full(4, -32768)
        if status_data["sec_isvalid"] == 0:
            updated_vector_magi = np.full(4, -32768)

        mago_calibration = l1d_calibration_dataset["URFTOORFO"]
        magi_calibration = l1d_calibration_dataset["URFTOORFI"]
        offsets = l1d_calibration_dataset["offsets"]

        mago_out = calibrate_and_offset_vectors(
            updated_vector_mago, mago_calibration, offsets, is_magi=False
        )
        magi_out = calibrate_and_offset_vectors(
            updated_vector_magi, magi_calibration, offsets, is_magi=True
        )

        # Convert to ECLIPJ2000 frame.
        mago_inertial_vector = transform_to_inertial(
            sc_spin_phase_rad.values,
            sc_inertial_right.values,
            sc_inertial_decline.values,
            attitude_time,
            time_data["primary_epoch"],
            mago_out,
            ValidFrames.MAGO.spice_frame,
        )
        magi_inertial_vector = transform_to_inertial(
            sc_spin_phase_rad.values,
            sc_inertial_right.values,
            sc_inertial_decline.values,
            attitude_time,
            time_data["secondary_epoch"],
            magi_out,
            ValidFrames.MAGI.spice_frame,
        )

        met = grouped_data["met"][(grouped_data["group"] == group).values]
        met_all.append(met)
        mago_times_all.append(time_data["primary_epoch"])
        mago_vectors_all.append(mago_inertial_vector)
        magi_vectors_all.append(magi_inertial_vector)
        magi_times_all.append(time_data["secondary_epoch"])

    if incomplete_groups:
        logger.info(
            f"The following mag groups were skipped due to "
            f"missing or duplicate pkt_counter values: "
            f"{incomplete_groups}"
        )

    mago_corrected, magnitude = apply_gradiometry_correction(
        np.array(mago_vectors_all),
        np.array(mago_times_all),
        np.array(magi_vectors_all),
        np.array(magi_times_all),
        l1d_calibration_dataset["gradiometer_factor"].squeeze(),
    )

    gse_vector, gsm_vector, rtn_vector = transform_to_frames(
        np.array(mago_times_all), mago_corrected
    )

    spherical = cartesian_to_spherical(gsm_vector)
    phi_gsm = spherical[:, 1]
    theta_gsm = spherical[:, 2]

    spherical = cartesian_to_spherical(gse_vector)
    phi_gse = spherical[:, 1]
    theta_gse = spherical[:, 2]

    # Omit the first value since we expect it to be extrapolated.
    for i in range(len(mago_corrected)):
        if i == 0:
            continue

        mag_data.append(
            _populate_instrument_header_items(met_all[i])
            | {
                "instrument": "mag",
                "mag_epoch": int(mago_times_all[i]),
                "mag_B_GSE": [Decimal(f"{v:.3f}") for v in gse_vector[i]],
                "mag_B_GSM": [Decimal(f"{v:.3f}") for v in gsm_vector[i]],
                "mag_B_RTN": [Decimal(f"{v:.3f}") for v in rtn_vector[i]],
                "mag_B_magnitude": Decimal(f"{magnitude[i]:.3f}"),
                "mag_phi_B_GSM": Decimal(f"{phi_gsm[i]:.3f}"),
                "mag_theta_B_GSM": Decimal(f"{theta_gsm[i]:.3f}"),
                "mag_phi_B_GSE": Decimal(f"{phi_gse[i]:.3f}"),
                "mag_theta_B_GSE": Decimal(f"{theta_gse[i]:.3f}"),
                "mag_hk_status": {
                    "hk1v5_warn": bool(status_data["hk1v5_warn"]),
                    "hk1v5_danger": bool(status_data["hk1v5_danger"]),
                    "hk1v5c_warn": bool(status_data["hk1v5c_warn"]),
                    "hk1v5c_danger": bool(status_data["hk1v5c_danger"]),
                    "hk1v8_warn": bool(status_data["hk1v8_warn"]),
                    "hk1v8_danger": bool(status_data["hk1v8_danger"]),
                    "hk1v8c_warn": bool(status_data["hk1v8c_warn"]),
                    "hk1v8c_danger": bool(status_data["hk1v8c_danger"]),
                    "fob_saturated": bool(status_data["fob_saturated"]),
                    "fib_saturated": bool(status_data["fib_saturated"]),
                    "mode": int(status_data["mode"]),
                    "icu_temp": int(status_data["icu_temp"]),
                    "hk2v5_warn": bool(status_data["hk2v5_warn"]),
                    "hk2v5_danger": bool(status_data["hk2v5_danger"]),
                    "hk2v5c_warn": bool(status_data["hk2v5c_warn"]),
                    "hk2v5c_danger": bool(status_data["hk2v5c_danger"]),
                    "hk3v3": int(status_data["hk3v3"]),
                    "hk3v3_current": int(status_data["hk3v3_current"]),
                    "pri_isvalid": bool(status_data["pri_isvalid"]),
                    "hkp8v5_warn": bool(status_data["hkp8v5_warn"]),
                    "hkp8v5_danger": bool(status_data["hkp8v5_danger"]),
                    "hkp8v5c_warn": bool(status_data["hkp8v5c_warn"]),
                    "hkp8v5c_danger": bool(status_data["hkp8v5c_danger"]),
                    "hkn8v5": int(status_data["hkn8v5"]),
                    "hkn8v5_current": int(status_data["hkn8v5_current"]),
                    "fob_temp": int(status_data["fob_temp"]),
                    "fib_temp": int(status_data["fib_temp"]),
                    "fob_range": int(status_data["fob_range"]),
                    "fib_range": int(status_data["fib_range"]),
                    "multbit_errs": bool(status_data["multbit_errs"]),
                    "sec_isvalid": bool(status_data["sec_isvalid"]),
                },
            }
        )

    return mag_data


def retrieve_matrix_from_single_l1b_calibration(
    calibration_dataset: xr.Dataset, is_mago: bool = True
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Retrieve the calibration matrix and time shift from the calibration dataset.

    Parameters
    ----------
    calibration_dataset : xarray.Dataset
        The calibration dataset containing the calibration matrices and time shift.
    is_mago : bool
        Whether the calibration is for mago or magi. If True, it retrieves the mago
        calibration matrix and time shift. If False, it retrieves the magi calibration
        matrix and time shift.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        The calibration matrix and time shift. These can be passed directly into
        update_vector, calibrate_vector, and shift_time.
    """
    if is_mago:
        calibration_matrix = calibration_dataset["MFOTOURFO"]
        time_shift = calibration_dataset["OTS"]
    else:
        calibration_matrix = calibration_dataset["MFITOURFI"]
        time_shift = calibration_dataset["ITS"]

    return calibration_matrix, time_shift
