"""Calculate Badtimes."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_badtimes(
    extendedspin_dataset: xr.Dataset,
    cullingmask_spins: NDArray,
    name: str,
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

    Returns
    -------
    badtimes_dataset : xarray.Dataset
        Dataset containing the extendedspin data that has been culled.
    """
    culled_spins = np.setdiff1d(
        extendedspin_dataset["spin_number"].values, cullingmask_spins
    )

    filtered_dataset = extendedspin_dataset.sel(spin_number=culled_spins)

    badtimes_dataset = create_dataset(filtered_dataset, name, "l1b")

    # Inherents coordinates from the filtered dataset if empty.
    if badtimes_dataset.dims["spin_number"] == 0:
        badtimes_dataset = badtimes_dataset.assign_coords(filtered_dataset.coords)

    return badtimes_dataset
