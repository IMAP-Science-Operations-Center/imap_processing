"""Calculates Histogram."""

import logging

logger = logging.getLogger(__name__)


def calculate_histogram(l1a_histograms_dict, l1b_cullingmask_dict):
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
    histogram_dict : dict
        Dictionary containing the data.
    """
    histogram_dict = {}

    # TODO: Use temporal culling mask for filtering

    histogram_dict["epoch"] = l1a_histograms_dict["epoch"]
    histogram_dict["sid"] = l1a_histograms_dict["sid"]
    histogram_dict["row"] = l1a_histograms_dict["row"]
    histogram_dict["column"] = l1a_histograms_dict["column"]
    histogram_dict["shcoarse"] = l1a_histograms_dict["shcoarse"]
    histogram_dict["spin"] = l1a_histograms_dict["spin"]
    histogram_dict["packetdata"] = l1a_histograms_dict["packetdata"]

    return histogram_dict
