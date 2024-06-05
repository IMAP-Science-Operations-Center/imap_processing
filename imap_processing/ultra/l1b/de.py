"""Calculates Annotated Direct Events."""

import numpy as np

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_de(data_dict, name):
    """
    Create dataset with defined datatypes for Direct Event Data.

    Parameters
    ----------
    data_dict: : dict
        L1a data dictionary.
    name: str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    de_dict = {}

    # Placeholder for calculations
    dataset = data_dict["imap_ultra_l1a_45sensor-de"]
    epoch = dataset.coords["epoch"].values

    de_dict["epoch"] = epoch
    de_dict["x_front"] = np.zeros(len(epoch), dtype=np.uint64)

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset
