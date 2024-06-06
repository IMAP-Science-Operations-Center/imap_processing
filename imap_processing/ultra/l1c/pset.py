"""Calculates Pointing Set Grids."""

import numpy as np

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_pset(data_dict, name):
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    data_dict: : dict
        L1b data dictionary.
    name: str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict = {}

    # Placeholder for calculations
    dataset = data_dict["imap_ultra_l1b_45sensor-de"]
    epoch = dataset.coords["epoch"].values

    pset_dict["epoch"] = epoch
    pset_dict["esa_step"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
