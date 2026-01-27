"""Calculate Annotated Direct Events."""

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import parse_filename_like
from imap_processing.quality_flags import (
    ImapDEOutliersUltraFlags,
    ImapDEScatteringUltraFlags,
)
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.repoint import get_pointing_times_from_id
from imap_processing.spice.time import (
    et_to_met,
)
from imap_processing.ultra.l1b.lookup_utils import get_geometric_factor
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_culling import flag_scattering
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    StopType,
    determine_ebin_pulse_height,
    determine_ebin_ssd,
    determine_species,
    get_coincidence_positions,
    get_ctof,
    get_de_energy_kev,
    get_de_velocity,
    get_efficiency,
    get_energy_pulse_height,
    get_energy_ssd,
    get_event_times,
    get_front_x_position,
    get_front_y_position,
    get_fwhm,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_phi_theta,
    get_spin_info,
    get_ssd_back_position_and_tof_offset,
    get_ssd_tof,
    is_back_tof_valid,
    is_coin_ph_valid,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

FILLVAL_UINT8 = 255
FILLVAL_UINT32 = 4294967295
FILLVAL_FLOAT32 = -1.0e31


def calculate_de(
    de_dataset: xr.Dataset, aux_dataset: xr.Dataset, name: str, ancillary_files: dict
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Direct Event Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        L1a dataset containing direct event data.
    aux_dataset : xarray.Dataset
        L1a dataset containing auxiliary data.
    name : str
        Name of the l1a dataset.
    ancillary_files : dict
        Ancillary files.

    Returns
    -------
    dataset : xarray.Dataset
        L1b de dataset.
    """
    de_dict = {}
    sensor = parse_filename_like(name)["sensor"][0:2]

    # Define epoch
    de_dict["epoch"] = de_dataset["epoch"].data

    repoint_id = de_dataset.attrs.get("Repointing", None)
    if repoint_id is not None:
        repoint_id = int(repoint_id.replace("repoint", ""))

    # Add already populated fields.
    keys = [
        "coincidence_type",
        "start_type",
        "event_type",
        "de_event_met",
        "phase_angle",
        "event_id",
    ]
    dataset_keys = [
        "coin_type",
        "start_type",
        "stop_type",
        "shcoarse",
        "phase_angle",
        "event_id",
    ]
    # Populate de_dict with existing fields from de_dataset
    de_dict.update(
        {
            key: de_dataset[dataset_key]
            for key, dataset_key in zip(keys, dataset_keys, strict=False)
        }
    )
    valid_mask = de_dataset["start_type"].data != FILLVAL_UINT8
    ph_mask = np.isin(
        de_dataset["stop_type"].data, [StopType.Top.value, StopType.Bottom.value]
    )
    ssd_mask = np.isin(de_dataset["stop_type"].data, [StopType.SSD.value])

    valid_indices = np.nonzero(valid_mask)[0]
    ph_indices = np.nonzero(valid_mask & ph_mask)[0]
    ssd_indices = np.nonzero(valid_mask & ssd_mask)[0]
    # Instantiate arrays
    xf = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    yf = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    xb = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    yb = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    xc = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    d = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float64)
    r = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    phi = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    theta = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    tof = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    etof = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    ctof = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    tof_energy = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    magnitude_v = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    energy = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    e_bin = np.full(len(de_dataset["epoch"]), FILLVAL_UINT8, dtype=np.uint8)
    e_bin_l1a = np.full(len(de_dataset["epoch"]), FILLVAL_UINT8, dtype=np.uint8)
    species_bin = np.full(len(de_dataset["epoch"]), FILLVAL_UINT8, dtype=np.uint8)
    t2 = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    event_times = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float64)
    spin_starts = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float64)
    shape = (len(de_dataset["epoch"]), 3)
    sc_velocity = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)
    sc_dps_velocity = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)
    helio_velocity = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)
    velocities = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)
    v_hat = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)
    r_hat = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)

    start_type = np.full(len(de_dataset["epoch"]), FILLVAL_UINT8, dtype=np.uint8)
    quality_flags = np.full(
        de_dataset["epoch"].shape, ImapDEOutliersUltraFlags.NONE.value, dtype=np.uint16
    )

    scattering_quality_flags = np.full(
        de_dataset["epoch"].shape,
        ImapDEScatteringUltraFlags.NONE.value,
        dtype=np.uint16,
    )

    xf[valid_indices] = get_front_x_position(
        de_dataset["start_type"].data[valid_indices],
        de_dataset["start_pos_tdc"].data[valid_indices],
        f"ultra{sensor}",
        ancillary_files,
    )

    start_type[valid_indices] = de_dataset["start_type"].data[valid_indices]
    spin_ds = get_spin_info(aux_dataset, de_dataset["shcoarse"].data)

    (event_times[valid_mask], spin_starts[valid_mask]) = get_event_times(
        aux_dataset,
        de_dataset["shcoarse"].data[valid_mask],
        de_dataset["phase_angle"].data[valid_mask],
        spin_ds.isel(epoch=valid_mask),
    )

    de_dict["spin"] = spin_ds.spin_number.data
    de_dict["event_times"] = event_times.astype(np.float64)
    # Pulse height
    ph_result = get_ph_tof_and_back_positions(
        de_dataset, xf, f"ultra{sensor}", ancillary_files
    )
    tof[ph_indices] = ph_result.tof
    t2[ph_indices] = ph_result.t2
    xb[ph_indices] = ph_result.xb
    yb[ph_indices] = ph_result.yb
    d[ph_indices], yf[ph_indices] = get_front_y_position(
        de_dataset["start_type"].data[ph_indices], yb[ph_indices], ancillary_files
    )
    energy[ph_indices], _ = get_energy_pulse_height(
        de_dataset["stop_type"].data[ph_indices],
        de_dataset["energy_ph"].data[ph_indices],
        xb[ph_indices],
        yb[ph_indices],
        f"ultra{sensor}",
        ancillary_files,
        quality_flags[ph_indices],
    )
    r[ph_indices] = get_path_length(
        (xf[ph_indices], yf[ph_indices]),
        (xb[ph_indices], yb[ph_indices]),
        d[ph_indices],
    )
    phi[ph_indices], theta[ph_indices] = get_phi_theta(
        (xf[ph_indices], yf[ph_indices]),
        (xb[ph_indices], yb[ph_indices]),
        d[ph_indices],
    )
    etof[ph_indices], xc[ph_indices] = get_coincidence_positions(
        de_dataset.isel(epoch=ph_indices),
        t2[ph_indices],
        f"ultra{sensor}",
        ancillary_files,
    )
    backtofvalid_quality_flags = np.zeros(len(ph_indices), dtype=quality_flags.dtype)
    backtofvalid, backtofvalid_quality_flags = is_back_tof_valid(
        de_dataset.isel(epoch=ph_indices),
        xf[ph_indices],
        f"ultra{sensor}",
        ancillary_files,
        backtofvalid_quality_flags,
    )

    coinphvalid_quality_flags = np.zeros(len(ph_indices), dtype=quality_flags.dtype)
    coinphvalid, coinphvalid_quality_flags = is_coin_ph_valid(
        etof[ph_indices],
        xc[ph_indices],
        xb[ph_indices],
        de_dataset["stop_north_tdc"][ph_indices].values,
        de_dataset["stop_south_tdc"][ph_indices].values,
        de_dataset["stop_east_tdc"][ph_indices].values,
        de_dataset["stop_west_tdc"][ph_indices].values,
        f"ultra{sensor}",
        ancillary_files,
        coinphvalid_quality_flags,
    )
    quality_flags[ph_indices] |= coinphvalid_quality_flags
    quality_flags[ph_indices] |= backtofvalid_quality_flags

    e_bin[ph_indices] = determine_ebin_pulse_height(
        energy[ph_indices],
        tof[ph_indices],
        r[ph_indices],
        backtofvalid,
        coinphvalid,
        ancillary_files,
    )
    species_bin[ph_indices] = determine_species(e_bin[ph_indices], "PH")
    ctof[ph_indices], magnitude_v[ph_indices] = get_ctof(
        tof[ph_indices], r[ph_indices], "PH"
    )

    # SSD
    tof[ssd_indices] = get_ssd_tof(de_dataset, xf, f"ultra{sensor}", ancillary_files)
    yb[ssd_indices], _, ssd_number = get_ssd_back_position_and_tof_offset(
        de_dataset, f"ultra{sensor}", ancillary_files
    )
    xc[ssd_indices] = np.zeros(len(ssd_indices))
    xb[ssd_indices] = np.zeros(len(ssd_indices))
    etof[ssd_indices] = np.zeros(len(ssd_indices))
    d[ssd_indices], yf[ssd_indices] = get_front_y_position(
        de_dataset["start_type"].data[ssd_indices], yb[ssd_indices], ancillary_files
    )
    energy[ssd_indices] = get_energy_ssd(de_dataset, ssd_number, ancillary_files)
    r[ssd_indices] = get_path_length(
        (xf[ssd_indices], yf[ssd_indices]),
        (xb[ssd_indices], yb[ssd_indices]),
        d[ssd_indices],
    )
    phi[ssd_indices], theta[ssd_indices] = get_phi_theta(
        (xf[ssd_indices], yf[ssd_indices]),
        (xb[ssd_indices], yb[ssd_indices]),
        d[ssd_indices],
    )
    e_bin[ssd_indices] = determine_ebin_ssd(
        energy[ssd_indices],
        tof[ssd_indices],
        r[ssd_indices],
        f"ultra{sensor}",
        ancillary_files,
    )
    species_bin[ssd_indices] = determine_species(e_bin[ssd_indices], "SSD")
    ctof[ssd_indices], magnitude_v[ssd_indices] = get_ctof(
        tof[ssd_indices], r[ssd_indices], "SSD"
    )

    # Combine ph_yb and ssd_yb along with their indices
    de_dict["x_front"] = xf.astype(np.float32)
    de_dict["event_times"] = event_times
    de_dict["spin_starts"] = spin_starts
    de_dict["y_front"] = yf
    de_dict["x_back"] = xb
    de_dict["y_back"] = yb
    de_dict["x_coin"] = xc
    de_dict["tof_start_stop"] = tof
    de_dict["tof_stop_coin"] = etof
    de_dict["tof_corrected"] = ctof
    de_dict["velocity_magnitude"] = magnitude_v
    de_dict["front_back_distance"] = d
    de_dict["path_length"] = r
    de_dict["phi"] = phi
    de_dict["theta"] = theta

    velocities[valid_indices], v_hat[valid_indices], r_hat[valid_indices] = (
        get_de_velocity(
            (de_dict["x_front"][valid_indices], de_dict["y_front"][valid_indices]),
            (de_dict["x_back"][valid_indices], de_dict["y_back"][valid_indices]),
            de_dict["front_back_distance"][valid_indices],
            de_dict["tof_start_stop"][valid_indices],
        )
    )
    de_dict["direct_event_unit_velocity"] = v_hat.astype(np.float32)
    de_dict["direct_event_unit_position"] = r_hat.astype(np.float32)

    tof_energy[valid_indices] = get_de_energy_kev(
        velocities[valid_indices],
        species_bin[valid_indices],
        quality_flags[valid_indices],
    )
    de_dict["tof_energy"] = tof_energy
    de_dict["energy"] = energy
    de_dict["computed_ebin"] = e_bin
    valid_ebin = de_dataset["bin"].values != FILLVAL_UINT32
    e_bin_l1a[valid_ebin] = de_dataset["bin"].values[valid_ebin]
    de_dict["ebin"] = e_bin_l1a
    de_dict["species"] = species_bin

    # Annotated Events.
    ultra_frame = getattr(SpiceFrame, f"IMAP_ULTRA_{sensor}")

    # Account for counts=0 (event times have FILL value)
    valid_events = (event_times != FILLVAL_FLOAT32).copy()
    if repoint_id is not None:
        # Check all valid event times to see which are in the pointing
        in_pointing = calculate_events_in_pointing(
            repoint_id, et_to_met(event_times[valid_events])
        )
        # Initialize an array of all events as False
        events_to_flag = np.zeros(len(quality_flags), dtype=bool)
        # Identify valid events that are outside the pointing
        events_to_flag[valid_events] = ~in_pointing
        # Update quality flags for valid events that are not in the pointing
        quality_flags[events_to_flag] |= ImapDEOutliersUltraFlags.DURINGREPOINT.value
        # Update valid_events to only include times within a pointing
        valid_events[valid_events] &= in_pointing

    (
        sc_velocity[valid_events],
        sc_dps_velocity[valid_events],
        helio_velocity[valid_events],
    ) = get_annotated_particle_velocity(
        event_times[valid_events],
        velocities.astype(np.float32)[valid_events],
        ultra_frame,
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
    )

    de_dict["velocity_sc"] = sc_velocity
    de_dict["velocity_dps_sc"] = sc_dps_velocity
    de_dict["velocity_dps_helio"] = helio_velocity

    de_dict["energy_spacecraft"] = get_de_energy_kev(
        sc_dps_velocity, species_bin, quality_flags
    )
    de_dict["energy_heliosphere"] = get_de_energy_kev(
        helio_velocity, species_bin, quality_flags
    )

    de_dict["phi_fwhm"], de_dict["theta_fwhm"] = get_fwhm(
        start_type,
        f"ultra{sensor}",
        de_dict["tof_energy"],
        de_dict["phi"],
        de_dict["theta"],
        ancillary_files,
    )
    de_dict["event_efficiency"] = get_efficiency(
        de_dict["tof_energy"], de_dict["phi"], de_dict["theta"], ancillary_files
    )
    de_dict["geometric_factor_blades"] = get_geometric_factor(
        de_dict["phi"],
        de_dict["theta"],
        quality_flags,
        ancillary_files,
        "l1b-sensor-gf-blades",
    )
    de_dict["geometric_factor_noblades"] = get_geometric_factor(
        de_dict["phi"],
        de_dict["theta"],
        quality_flags,
        ancillary_files,
        "l1b-sensor-gf-noblades",
    )

    de_dict["quality_outliers"] = quality_flags
    flag_scattering(
        de_dict["tof_energy"],
        de_dict["theta"],
        de_dict["phi"],
        ancillary_files,
        sensor,
        scattering_quality_flags,
    )
    de_dict["quality_scattering"] = scattering_quality_flags

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset


def calculate_events_in_pointing(
    repoint_id: int,
    event_times: np.ndarray,
) -> np.ndarray:
    """
    Calculate boolean array of events within a pointing.

    Parameters
    ----------
    repoint_id : int
        The repointing ID.
    event_times : np.ndarray
        Array of event times in MET.

    Returns
    -------
    in_pointing : np.ndarray
        Boolean array indicating whether each event is within the pointing period
        combined with the valid_events mask.
    """
    pointing_start_met, pointing_end_met = get_pointing_times_from_id(repoint_id)
    # Check which events are within the pointing
    in_pointing = (event_times >= pointing_start_met) & (
        event_times <= pointing_end_met
    )
    return in_pointing
