"""Calculates Pointing Set Grids."""

import numpy as np


def calculate_pset_exposure(l1b_de_dict, l1b_extendedspin_dict, l1b_cullingmask_dict):
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    l1b_de_dict: : dict
        L1b data dictionary.
    l1b_extendedspin_dict: : dict
        L1b data dictionary.
    l1b_cullingmask_dict: : dict
        L1b data dictionary.

    Returns
    -------
    exposure_dict : dict
        Dictionary containing the data.
    """
    exposure_dict = {}

    # Exposure times
    exposure_dict["epoch"] = l1b_de_dict["epoch"]
    exposure_dict["spin_angle"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    exposure_dict["esa_step"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    exposure_dict["time_bin"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    exposure_dict["species_bin"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    exposure_dict["exposure_times"] = np.zeros((len(l1b_de_dict["epoch"]), 1))

    return exposure_dict


def calculate_pset_sensitivity(
    l1b_de_dict, l1b_extendedspin_dict, l1b_cullingmask_dict
):
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    l1b_de_dict: : dict
        L1b data dictionary.
    l1b_extendedspin_dict: : dict
        L1b data dictionary.
    l1b_cullingmask_dict: : dict
        L1b data dictionary.

    Returns
    -------
    sensitivity_dict : dict
        Dictionary containing the data.
    """
    sensitivity_dict = {}

    # Sensitivity
    sensitivity_dict["epoch"] = l1b_de_dict["epoch"]
    sensitivity_dict["spin_angle"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    sensitivity_dict["esa_step"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    sensitivity_dict["time_bin"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    sensitivity_dict["species_bin"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    sensitivity_dict["sensitivity"] = np.zeros((len(l1b_de_dict["epoch"]), 1))

    return sensitivity_dict


def calculate_pset_counts(l1b_de_dict, l1b_extendedspin_dict, l1b_cullingmask_dict):
    """
    Create counts dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    l1b_de_dict: : dict
        L1b data dictionary.
    l1b_extendedspin_dict: : dict
        L1b data dictionary.
    l1b_cullingmask_dict: : dict
        L1b data dictionary.

    Returns
    -------
    counts_dict : dict
        Dictionary containing the data.
    """
    counts_dict = {}

    # Counts
    counts_dict["epoch"] = l1b_de_dict["epoch"]
    counts_dict["spin_angle"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    counts_dict["esa_step"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    counts_dict["time_bin"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    counts_dict["species_bin"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    counts_dict["counts"] = np.zeros((len(l1b_de_dict["epoch"]), 1))

    return counts_dict


def calculate_pset_backgroundrates(
    l1b_de_dict, l1b_extendedspin_dict, l1b_cullingmask_dict
):
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    l1b_de_dict: : dict
        L1b data dictionary.
    l1b_extendedspin_dict: : dict
        L1b data dictionary.
    l1b_cullingmask_dict: : dict
        L1b data dictionary.

    Returns
    -------
    backgroundrates_dict : dict
        Dictionary containing the data.
    """
    backgroundrates_dict = {}

    # Background Rates
    backgroundrates_dict["epoch"] = l1b_de_dict["epoch"]
    backgroundrates_dict["spin_angle"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    backgroundrates_dict["esa_step"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    backgroundrates_dict["time_bin"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    backgroundrates_dict["species_bin"] = np.zeros((len(l1b_de_dict["epoch"]), 1))
    backgroundrates_dict["background_rates"] = np.zeros((len(l1b_de_dict["epoch"]), 1))

    return backgroundrates_dict
