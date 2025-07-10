"""Functions to support I-ALiRT MAG packet parsing."""

import logging
from decimal import Decimal
from typing import Union

import numpy as np
import xarray as xr

from imap_processing.ialirt.l0.mag_l0_ialirt_data import (
    Packet0,
    Packet1,
    Packet2,
    Packet3,
)
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.mag.l1a.mag_l1a_data import TimeTuple
from imap_processing.mag.l1b.mag_l1b import (
    calibrate_vector,
    shift_time,
)
from imap_processing.spice.time import met_to_ttj2000ns, met_to_utc

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

    time_data: dict[str, Union[int, float]] = {
        "pri_coarsetm": int(pri_coarsetm.item()),
        "pri_fintm": int(pri_fintm.item()),
        "sec_coarsetm": int(sec_coarsetm.item()),
        "sec_fintm": int(sec_fintm.item()),
    }

    primary_time = TimeTuple(int(pri_coarsetm.item()), int(pri_fintm.item()))
    secondary_time = TimeTuple(int(sec_coarsetm.item()), int(sec_fintm.item()))
    time_data_pri_met = primary_time.to_seconds()
    time_data_primary_ttj2000ns = met_to_ttj2000ns(time_data_pri_met)
    time_data["primary_epoch"] = shift_time(
        time_data_primary_ttj2000ns, time_shift_mago
    )
    time_data_sec_met = secondary_time.to_seconds()
    time_data_secondary_ttj2000ns = met_to_ttj2000ns(time_data_sec_met)
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


def process_packet(
    accumulated_data: xr.Dataset, calibration_dataset: xr.Dataset
) -> tuple[list[dict], list[dict]]:
    """
    Parse the MAG packets.

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.
    calibration_dataset : xr.Dataset
        Calibration dataset.

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

    grouped_data = find_groups(accumulated_data, (0, 3), "pkt_counter", "met")

    unique_groups = np.unique(grouped_data["group"])
    l1b_data = []
    mag_data = []

    for group in unique_groups:
        # Get status values for each group.
        status_values = grouped_data["mag_status"][
            (grouped_data["group"] == group).values
        ]
        pkt_counter = grouped_data["pkt_counter"][
            (grouped_data["group"] == group).values
        ]

        if not np.array_equal(pkt_counter, np.arange(4)):
            logger.info(
                f"Group {group} does not contain all values from 0 to "
                f"3 without duplicates."
            )
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
            calibration_dataset,
        )

        # Note: primary = MAGo, secondary = MAGi.
        # Populate with a FILL value if either sensor is invalid,
        # but not both.
        if status_data["pri_isvalid"] == 0:
            updated_vector_mago = np.full(4, -32768)
        if status_data["sec_isvalid"] == 0:
            updated_vector_magi = np.full(4, -32768)

        science_data.update(
            {
                "calibrated_pri_x": updated_vector_mago[0],
                "calibrated_pri_y": updated_vector_mago[1],
                "calibrated_pri_z": updated_vector_mago[2],
                "calibrated_sec_x": updated_vector_magi[0],
                "calibrated_sec_y": updated_vector_magi[1],
                "calibrated_sec_z": updated_vector_magi[2],
            }
        )

        l1b_data.append({**status_data, **science_data, **time_data})

        # Placeholder for real data.
        met = grouped_data["met"][(grouped_data["group"] == group).values]
        mag_data.append(
            {
                "apid": 478,
                "met": int(met.values.min()),
                "met_in_utc": met_to_utc(met.values.min()).split(".")[0],
                "ttj2000ns": int(met_to_ttj2000ns(met.values.min())),
                "mag_4s_b_gse": [Decimal("0.0") for _ in range(3)],
                "mag_4s_b_gsm": [Decimal("0.0") for _ in range(3)],
                "mag_4s_b_rtn": [Decimal("0.0") for _ in range(3)],
                "mag_phi_4s_b_gsm": Decimal("0.0"),
                "mag_theta_4s_b_gsm": Decimal("0.0"),
            }
        )

    return mag_data, l1b_data


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
