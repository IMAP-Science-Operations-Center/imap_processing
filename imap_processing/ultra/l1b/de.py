"""Calculate Annotated Direct Events."""

import numpy as np
import xarray as xr

from imap_processing.ultra.l1b.ultra_l1b_extended import (
    StopType,
    determine_species_pulse_height,
    determine_species_ssd,
    get_coincidence_positions,
    get_ctof,
    get_energy_pulse_height,
    get_energy_ssd,
    get_front_x_position,
    get_front_y_position,
    get_particle_velocity,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_ssd_back_position_and_tof_offset,
    get_ssd_tof,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_de(de_dataset: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Direct Event Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Dataset containing direct event data.
    name : str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    de_dict = {}
    sensor = name.split("_l1b_")[1].split("sensor-de")[0]

    # Drop events with invalid start type.
    de_dataset = de_dataset.where(
        de_dataset["START_TYPE"] != np.iinfo(np.int64).min, drop=True
    )
    # Define epoch.
    de_dict["epoch"] = de_dataset["epoch"]

    xf = get_front_x_position(
        de_dataset["START_TYPE"].data,
        de_dataset["START_POS_TDC"].data,
    )

    # Pulse height
    ph_indices = np.nonzero(
        np.isin(de_dataset["STOP_TYPE"], [StopType.Top.value, StopType.Bottom.value])
    )[0]
    ph_tof, ph_t2, ph_xb, ph_yb = get_ph_tof_and_back_positions(
        de_dataset, xf, f"ultra{sensor}"
    )
    ph_d, ph_yf = get_front_y_position(de_dataset["START_TYPE"].data[ph_indices], ph_yb)
    ph_energy = get_energy_pulse_height(
        de_dataset["STOP_TYPE"].data[ph_indices],
        de_dataset["ENERGY_PH"].data[ph_indices],
        ph_xb,
        ph_yb,
    )
    ph_r = get_path_length((xf[ph_indices], ph_yf), (ph_xb, ph_yb), ph_d)
    ph_bin = determine_species_pulse_height(ph_energy, ph_tof, ph_r)
    ph_etof, ph_xc = get_coincidence_positions(
        de_dataset.isel(epoch=ph_indices), ph_t2, f"ultra{sensor}"
    )
    ph_ctof = get_ctof(ph_tof, ph_r, "PH")

    # SSD
    ssd_indices = np.nonzero(np.isin(de_dataset["STOP_TYPE"], StopType.SSD.value))[0]
    ssd_tof = get_ssd_tof(de_dataset, xf)
    ssd_yb, ssd_tof_offset, ssd_number = get_ssd_back_position_and_tof_offset(
        de_dataset
    )
    ssd_xb = np.zeros(len(ssd_yb))
    ssd_d, ssd_yf = get_front_y_position(
        de_dataset["START_TYPE"].data[ssd_indices], ssd_yb
    )
    ssd_energy = get_energy_ssd(de_dataset, ssd_number)
    ssd_r = get_path_length((xf[ssd_indices], ssd_yf), (ssd_xb, ssd_yb), ssd_d)
    ssd_bin = determine_species_ssd(
        ssd_energy,
        ssd_tof,
        ssd_r,
    )
    ssd_ctof = get_ctof(ssd_tof, ssd_r, "SSD")

    # Combine ph_yb and ssd_yb along with their indices
    combined_indices = np.argsort(np.concatenate((ph_indices, ssd_indices)))
    de_dict["x_front"] = xf
    yb = np.concatenate((ph_yb, ssd_yb))
    de_dict["y_back"] = yb[combined_indices]
    xb = np.concatenate((ph_xb, ssd_xb))
    de_dict["x_back"] = xb[combined_indices]
    xcoin = np.concatenate((ph_xc, np.zeros(len(ssd_indices))))
    de_dict["x_coin"] = xcoin[combined_indices]
    yf = np.concatenate((ph_yf, ssd_yf))
    de_dict["y_front"] = yf[combined_indices]
    d = np.concatenate((ph_d, ssd_d))
    de_dict["front_back_distance"] = d[combined_indices]
    r = np.concatenate((ph_r, ssd_r))
    de_dict["path_length"] = r[combined_indices]
    tof = np.concatenate((ph_tof, ssd_tof))
    de_dict["tof_start_stop"] = tof[combined_indices]
    etof = np.concatenate((ph_etof, np.zeros(len(ssd_indices))))
    de_dict["tof_stop_coin"] = etof[combined_indices]

    ctof = np.concatenate((ph_ctof, ssd_ctof))
    de_dict["tof_corrected"] = ctof[combined_indices]

    keys = [
        "coincidence_type",
        "start_type",
        "event_type",
        "de_event_met",
    ]
    dataset_keys = ["COIN_TYPE", "START_TYPE", "STOP_TYPE", "SHCOARSE"]

    de_dict.update(
        {key: de_dataset[dataset_key] for key, dataset_key in zip(keys, dataset_keys)}
    )

    vx_ultra, vy_ultra, vz_ultra = get_particle_velocity(
        (de_dict["x_front"], de_dict["y_front"]),
        (de_dict["x_back"], de_dict["y_back"]),
        de_dict["front_back_distance"],
        de_dict["tof_start_stop"],
    )

    # We need to fill velocities that have negative tof values.
    de_dict["vx_ultra"] = np.full_like(
        vx_ultra, np.finfo(np.float64).min, dtype=np.float64
    )
    de_dict["vy_ultra"] = np.full_like(
        vy_ultra, np.finfo(np.float64).min, dtype=np.float64
    )
    de_dict["vz_ultra"] = np.full_like(
        vz_ultra, np.finfo(np.float64).min, dtype=np.float64
    )

    condition = de_dict["tof_start_stop"] > 0
    de_dict["vx_ultra"][condition] = vx_ultra[condition]
    de_dict["vy_ultra"][condition] = vy_ultra[condition]
    de_dict["vz_ultra"][condition] = vz_ultra[condition]

    energy = np.concatenate((ph_energy, ssd_energy))
    de_dict["energy"] = energy[combined_indices]

    species = np.concatenate((ph_bin, ssd_bin))
    de_dict["species"] = species[combined_indices]

    # Annotated Events.
    # TODO: since the pointing (dps) frame is not for this timerange this will not work.
    # position = np.stack(
    #     (de_dict["vx_ultra"], de_dict["vy_ultra"], de_dict["vz_ultra"]), axis=-1
    # )
    #
    # ultra_frame = getattr(SpiceFrame, f"IMAP_ULTRA_{sensor}")
    # sc_velocity, sc_dps_velocity, helio_velocity = get_annotated_particle_velocity(
    #     de_dataset.data_vars["EVENTTIMES"],
    #     position,
    #     ultra_frame,
    #     SpiceFrame.IMAP_DPS,
    #     SpiceFrame.IMAP_SPACECRAFT,
    # )
    # TODO: this is a temporary fix.
    sc_velocity = np.zeros((len(de_dict["epoch"]), 3))
    sc_dps_velocity = np.zeros((len(de_dict["epoch"]), 3))
    helio_velocity = np.zeros((len(de_dict["epoch"]), 3))

    de_dict["vx_sc"], de_dict["vy_sc"], de_dict["vz_sc"] = (
        sc_velocity[:, 0],
        sc_velocity[:, 1],
        sc_velocity[:, 2],
    )
    de_dict["vx_dps_sc"], de_dict["vy_dps_sc"], de_dict["vz_dps_sc"] = (
        sc_dps_velocity[:, 0],
        sc_dps_velocity[:, 1],
        sc_dps_velocity[:, 2],
    )
    de_dict["vx_dps_helio"], de_dict["vy_dps_helio"], de_dict["vz_dps_helio"] = (
        helio_velocity[:, 0],
        helio_velocity[:, 1],
        helio_velocity[:, 2],
    )

    # TODO: TBD.
    de_dict["event_efficiency"] = np.zeros(len(de_dict["epoch"]), dtype=np.float64)

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset
