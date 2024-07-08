"""Calculate Pointing Set Grids."""

import numpy as np
import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_pset(pset_dataset: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    pset_dataset : xarray.Dataset
        Dataset containing histogram data.
    name : str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict = {}

    # Placeholder for calculations
    # TODO: come back and update this data structure.
    epoch = pset_dataset.coords["epoch"].values

    pset_dict["epoch"] = epoch
    pset_dict["esa_step"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
