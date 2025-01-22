"""Calculate Badtimes."""

import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_badtimes(extended_spin_dict: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Badtimes Data.

    Parameters
    ----------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    name : str
        Name of the dataset.

    Returns
    -------
    badtimes_dataset : xarray.Dataset
        Dataset containing the data.
    """
    badtimes_dataset = create_dataset(extended_spin_dict, name, "l1b")

    return badtimes_dataset
