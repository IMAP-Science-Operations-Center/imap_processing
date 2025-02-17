"""Calculate Badtimes."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_badtimes(
    extendedspin_dataset: xr.Dataset,
    cullingmask_spins: NDArray,
    name: str,
    data_version: str,
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Badtimes Data.

    Parameters
    ----------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    cullingmask_spins : np.ndarray
        Dataset containing the culled data.
    name : str
        Name of the dataset.
    data_version : str
        Version of the data.

    Returns
    -------
    badtimes_dataset : xarray.Dataset
        Dataset containing the data.
    """
    spins = np.setdiff1d(extendedspin_dataset["spin_number"].values, cullingmask_spins)

    filtered_dataset = extendedspin_dataset.sel(spin_number=spins)

    dataset_dict = {
        **{var: filtered_dataset[var].values for var in filtered_dataset.data_vars},
        **{coord: filtered_dataset[coord].values for coord in filtered_dataset.coords},
    }

    badtimes_dataset = create_dataset(dataset_dict, name, "l1b", data_version)

    return badtimes_dataset
