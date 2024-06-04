"""Calculates Extended Spin."""

import numpy as np

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_extendedspin(data_dict, name):
    """
    Create dataset with defined datatypes for Extended Spin Data.

    Parameters
    ----------
    data_dict: : dict
        L1a data dictionary.
    name: str
        Name of the dataset.

    Returns
    -------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    """
    extendedspin_dict = {}

    dataset = data_dict["imap_ultra_l1a_45sensor-rates"]
    epoch = dataset.coords["epoch"].values

    # Placeholder for calculations
    extendedspin_dict["epoch"] = epoch
    extendedspin_dict["spin_number"] = np.zeros(len(epoch), dtype=np.uint64)

    extendedspin_dataset = create_dataset(extendedspin_dict, name, "l1b")

    return extendedspin_dataset
