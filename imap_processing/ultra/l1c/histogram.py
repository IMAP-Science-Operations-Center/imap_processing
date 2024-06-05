"""Calculates Histogram."""

import numpy as np

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_histogram(l1a_histogram_dict, l1b_cullingmask_dict, name):
    """
    Create dictionary with defined datatype for Histogram Data.

    Parameters
    ----------
    l1a_histogram_dict: : dict
        L1a data dictionary.
    l1b_cullingmask_dict: : dict
        L1b data dictionary.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    histogram_dict = {}

    # Placeholder for calculations
    dataset = l1a_histogram_dict["imap_ultra_l1a_45sensor-histogram"]
    epoch = dataset.coords["epoch"].values

    histogram_dict["epoch"] = epoch
    histogram_dict["sid"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(histogram_dict, name, "l1c")

    return dataset
