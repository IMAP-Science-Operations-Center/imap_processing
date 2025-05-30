"""Calculate Annotated Direct Events."""

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import parse_filename_like
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    StopType,
    determine_species,
    get_coincidence_positions,
    get_ctof,
    get_de_energy_kev,
    get_de_velocity,
    get_efficiency,
    get_energy_pulse_height,
    get_energy_ssd,
    get_eventtimes,
    get_front_x_position,
    get_front_y_position,
    get_fwhm,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_phi_theta,
    get_ssd_back_position_and_tof_offset,
    get_ssd_tof,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

FILLVAL_UINT8 = 255
FILLVAL_FLOAT32 = -1.0e31


def calculate_de(
    de_dataset: xr.Dataset, name: str, ancillary_files: dict
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Direct Event Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        L1a dataset containing direct event data.
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

    # Define epoch and spin.
    de_dict["epoch"] = de_dataset["epoch"].data
    de_dict["spin"] = de_dataset["spin"].data

    # Add already populated fields.
    keys = [
        "coincidence_type",
        "start_type",
        "event_type",
        "de_event_met",
        "phase_angle",
        "spin",
    ]
    dataset_keys = [
        "coin_type",
        "start_type",
        "stop_type",
        "shcoarse",
        "phase_angle",
        "spin",
    ]

    de_dict.update(
        {key: de_dataset[dataset_key] for key, dataset_key in zip(keys, dataset_keys)}
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
    magnitude_v = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    energy = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    species_bin = np.full(len(de_dataset["epoch"]), FILLVAL_UINT8, dtype=np.uint8)
    t2 = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    event_times = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float32)
    shape = (len(de_dataset["epoch"]), 3)
    sc_velocity = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)
    sc_dps_velocity = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)
    helio_velocity = np.full(shape, FILLVAL_FLOAT32, dtype=np.float32)
    spin_starts = np.full(len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float64)
    spin_period_sec = np.full(
        len(de_dataset["epoch"]), FILLVAL_FLOAT32, dtype=np.float64
    )
    start_type = np.full(len(de_dataset["epoch"]), FILLVAL_UINT8, dtype=np.uint8)

    xf[valid_indices] = get_front_x_position(
        de_dataset["start_type"].data[valid_indices],
        de_dataset["start_pos_tdc"].data[valid_indices],
        f"ultra{sensor}",
    )
    start_type[valid_indices] = de_dataset["start_type"].data[valid_indices]

    (
        event_times[valid_indices],
        spin_starts[valid_indices],
        spin_period_sec[valid_indices],
    ) = get_eventtimes(
        de_dataset["spin"].data[valid_indices],
        de_dataset["phase_angle"].data[valid_indices],
    )

    # Pulse height
    tof[ph_indices], t2[ph_indices], xb[ph_indices], yb[ph_indices] = (
        get_ph_tof_and_back_positions(de_dataset, xf, f"ultra{sensor}")
    )
    d[ph_indices], yf[ph_indices] = get_front_y_position(
        de_dataset["start_type"].data[ph_indices], yb[ph_indices]
    )
    energy[ph_indices] = get_energy_pulse_height(
        de_dataset["stop_type"].data[ph_indices],
        de_dataset["energy_ph"].data[ph_indices],
        xb[ph_indices],
        yb[ph_indices],
        f"ultra{sensor}",
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
    species_bin[ph_indices] = determine_species(tof[ph_indices], r[ph_indices], "PH")
    etof[ph_indices], xc[ph_indices] = get_coincidence_positions(
        de_dataset.isel(epoch=ph_indices), t2[ph_indices], f"ultra{sensor}"
    )
    ctof[ph_indices], magnitude_v[ph_indices] = get_ctof(
        tof[ph_indices], r[ph_indices], "PH"
    )

    # SSD
    tof[ssd_indices] = get_ssd_tof(de_dataset, xf, f"ultra{sensor}")
    yb[ssd_indices], _, ssd_number = get_ssd_back_position_and_tof_offset(
        de_dataset, f"ultra{sensor}"
    )
    xc[ssd_indices] = np.zeros(len(ssd_indices))
    xb[ssd_indices] = np.zeros(len(ssd_indices))
    etof[ssd_indices] = np.zeros(len(ssd_indices))
    d[ssd_indices], yf[ssd_indices] = get_front_y_position(
        de_dataset["start_type"].data[ssd_indices], yb[ssd_indices]
    )
    energy[ssd_indices] = get_energy_ssd(de_dataset, ssd_number)
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
    species_bin[ssd_indices] = determine_species(
        tof[ssd_indices], r[ssd_indices], "SSD"
    )
    ctof[ssd_indices], magnitude_v[ssd_indices] = get_ctof(
        tof[ssd_indices], r[ssd_indices], "SSD"
    )

    # Combine ph_yb and ssd_yb along with their indices
    de_dict["x_front"] = xf.astype(np.float32)
    de_dict["event_times"] = event_times
    de_dict["spin_starts"] = spin_starts
    de_dict["spin_period"] = spin_period_sec
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

    v, vhat, r = get_de_velocity(
        (de_dict["x_front"], de_dict["y_front"]),
        (de_dict["x_back"], de_dict["y_back"]),
        de_dict["front_back_distance"],
        de_dict["tof_start_stop"],
    )
    de_dict["direct_event_velocity"] = v.astype(np.float32)
    de_dict["direct_event_unit_velocity"] = vhat.astype(np.float32)
    de_dict["direct_event_unit_position"] = r.astype(np.float32)

    de_dict["tof_energy"] = get_de_energy_kev(v, species_bin)
    de_dict["energy"] = energy
    de_dict["species"] = species_bin

    # Annotated Events.
    ultra_frame = getattr(SpiceFrame, f"IMAP_ULTRA_{sensor}")

    # Account for counts=0 (event times have FILL value)
    valid_events = event_times != FILLVAL_FLOAT32
    if np.any(valid_events):
        (
            sc_velocity[valid_events],
            sc_dps_velocity[valid_events],
            helio_velocity[valid_events],
        ) = get_annotated_particle_velocity(
            event_times[valid_events],
            de_dict["direct_event_velocity"][valid_events],
            ultra_frame,
            SpiceFrame.IMAP_DPS,
            SpiceFrame.IMAP_SPACECRAFT,
        )

    de_dict["velocity_sc"] = sc_velocity
    de_dict["velocity_dps_sc"] = sc_dps_velocity
    de_dict["velocity_dps_helio"] = helio_velocity

    de_dict["energy_spacecraft"] = get_de_energy_kev(sc_dps_velocity, species_bin)
    de_dict["energy_heliosphere"] = get_de_energy_kev(helio_velocity, species_bin)

    de_dict["phi_fwhm"], de_dict["theta_fwhm"] = get_fwhm(
        start_type,
        f"ultra{sensor}",
        de_dict["tof_energy"],
        de_dict["phi"],
        de_dict["theta"],
    )
    de_dict["event_efficiency"] = get_efficiency(
        de_dict["tof_energy"], de_dict["phi"], de_dict["theta"], ancillary_files
    )

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset
