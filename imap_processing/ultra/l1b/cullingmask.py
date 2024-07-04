"""Calculate Culling Mask."""

import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_cullingmask(extended_spin_dict: dict, name: str) -> xr.Dataset:
    """
    Create dataset with defined datatype for Culling Mask Data.

    Parameters
    ----------
    extended_spin_dict : dict
        L1b data dictionary.
    name : str
        Name of the dataset.

    Returns
    -------
    cullingmask_dataset : xarray.Dataset
        Dataset containing the data.
    """
    cullingmask_dataset = create_dataset(extended_spin_dict, name, "l1b")

    return cullingmask_dataset
