"""Calculate Annotated Direct Events."""

import numpy as np
import xarray as xr

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

    # Placeholder for calculations
    epoch = de_dataset.coords["epoch"].values

    de_dict["epoch"] = de_dataset["epoch"]
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
    de_dict["energy"] = np.zeros(len(epoch), dtype=np.float64)
    de_dict["species"] = np.zeros(len(epoch), dtype=np.uint64)
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
