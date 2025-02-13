"""Calculate Badtimes."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray


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
    cullingmask_spins : NDArray
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

    badtimes_dataset = extendedspin_dataset.sel(spin_number=spins)

    return badtimes_dataset
