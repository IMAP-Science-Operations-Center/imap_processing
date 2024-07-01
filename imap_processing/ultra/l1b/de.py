"""Calculate Annotated Direct Events."""

import numpy as np
import xarray as xr

from imap_processing.ultra.l1b.ultra_l1b_extended import (
    determine_species_pulse_height,
    get_coincidence_positions,
    get_energy_pulse_height,
    get_front_x_position,
    get_front_y_position,
    get_particle_velocity,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_ssd_offset_and_positions,
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

    epoch = de_dataset.coords["epoch"].values

    de_dict["epoch"] = de_dataset["epoch"]

    # Pulse height
    ph_indices = np.where(
        (de_dataset["STOP_TYPE"] == 1) | (de_dataset["STOP_TYPE"] == 2)
    )
    ph_xf = get_front_x_position(
        de_dataset["START_TYPE"].data[ph_indices],
        de_dataset["START_POS_TDC"].data[ph_indices],
    )
    ph_tof, ph_t2, ph_xb, ph_yb = get_ph_tof_and_back_positions(de_dataset, ph_xf)
    ph_d, ph_yf = get_front_y_position(de_dataset[ph_indices], ph_yb)
    energy = get_energy_pulse_height(de_dataset, ph_xb, ph_yb)

    r = get_path_length((ph_xf, ph_yf), (ph_xb, ph_yb), ph_d)

    ctof, bin = determine_species_pulse_height(energy, ph_tof, r)

    vhat_x, vhat_y, vhat_z = get_particle_velocity(
        (ph_xf, ph_yf),
        (ph_xb, ph_yb),
        ph_d[ph_indices],
        ph_tof,
    )

    etof, xc = get_coincidence_positions(de_dataset, ph_tof)

    # SSD
    ssd_indices = np.where(de_dataset["STOP_TYPE"] >= 8)
    ssd_xf = get_front_x_position(
        de_dataset["START_TYPE"].data[ssd_indices],
        de_dataset["START_POS_TDC"].data[ssd_indices],
    )
    _, ssd_y_back, _ = get_ssd_offset_and_positions(de_dataset)
    ssd_d, ssd_yf = get_front_y_position(de_dataset[ssd_indices], ph_yb)
    ssd_indices, tof = get_ssd_tof(ssd_indices, de_dataset, ssd_xf)

    de_dict["x_front"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["y_front"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["x_back"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["y_back"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["x_coin"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["tof_start_stop"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["tof_stop_coin"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["tof_corrected"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["eventtype"] = np.zeros(len(epoch), dtype=np.uint64)
    de_dict["vx_ultra"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vy_ultra"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vz_ultra"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["etof"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["energy"] = np.zeros(len(epoch), dtype=np.uint64)
    de_dict["species"] = np.zeros(len(epoch), dtype=np.uint64)

    # TODO: add more fields in the future
    de_dict["event_efficiency"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vx_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vy_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vz_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vx_dps_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vy_dps_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vz_dps_sc"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vx_dps_helio"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vy_dps_helio"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["vz_dps_helio"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["eventtimes"] = np.zeros(len(epoch), dtype=np.float64)

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset
