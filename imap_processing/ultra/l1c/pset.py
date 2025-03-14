"""Calculate Pointing Set Grids."""

import numpy as np
import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_pset(
    de_dataset: xr.Dataset, name: str, data_version: str
) -> xr.Dataset:
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Dataset containing de data.
    extendedspin_dataset : xarray.Dataset
        Dataset containing extendedspin data.
    cullingmask_dataset : xarray.Dataset
        Dataset containing cullingmask data.
    name : str
        Name of the dataset.
    data_version : str
        Version of the data.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict = {}

    # Placeholder for calculations
    # TODO: come back and update this data structure.
    epoch = de_dataset.coords["epoch"].values

    # TODO: Add below.
    # intervals, energy_midpoints = build_energy_bins()
    # counts, latitude, longitude, healpix_number = get_spacecraft_histogram(
    #     de_dataset["velocity_dps_sc"].values,
    #     de_dataset["tof_energy"].values,
    #     intervals,
    #     nside=128,
    # )
    #
    # background_rates = get_background_rates()

    pset_dict["epoch"] = epoch
    pset_dict["esa_step"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(pset_dict, name, "l1c", data_version)

    return dataset
