"""Calculates Histogram."""

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_histogram(l1a_histograms_dict, l1b_cullingmask_dict, name):
    """
    Create dictionary with defined datatype for Histogram Data.

    Parameters
    ----------
    l1a_histograms_dict: : dict
        L1a data dictionary.
    l1b_cullingmask_dict: : dict
        L1b data dictionary.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    dataset = create_dataset(l1a_histograms_dict, name, "l1c")

    return dataset
