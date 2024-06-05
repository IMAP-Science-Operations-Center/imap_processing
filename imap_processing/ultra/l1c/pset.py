"""Calculates Pointing Set Grids."""

import numpy as np

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_pset_exposure(
    l1b_de_dict, l1b_extendedspin_dict, l1b_cullingmask_dict, name
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
    name: str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict = {}

    # Placeholder for calculations
    dataset = l1b_de_dict["imap_ultra_l1a_45sensor-de"]
    epoch = dataset.coords["epoch"].values

    pset_dict["epoch"] = epoch
    pset_dict["esa_step"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset


def calculate_pset_sensitivity(
    l1b_de_dict, l1b_extendedspin_dict, l1b_cullingmask_dict, name
):
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    l1b_de_dict : dict
        L1b data dictionary.
    l1b_extendedspin_dict : dict
        L1b data dictionary.
    l1b_cullingmask_dict : dict
        L1b data dictionary.
    name : str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict = {}

    # Placeholder for calculations
    dataset = l1b_de_dict["imap_ultra_l1a_45sensor-de"]
    epoch = dataset.coords["epoch"].values

    pset_dict["epoch"] = epoch
    pset_dict["esa_step"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset


def calculate_pset_counts(
    l1b_de_dict, l1b_extendedspin_dict, l1b_cullingmask_dict, name
):
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
    name: str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict = {}

    # Placeholder for calculations
    dataset = l1b_de_dict["imap_ultra_l1a_45sensor-de"]
    epoch = dataset.coords["epoch"].values

    pset_dict["epoch"] = epoch
    pset_dict["esa_step"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset


def calculate_pset_backgroundrates(
    l1b_de_dict, l1b_extendedspin_dict, l1b_cullingmask_dict, name
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
    name: str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict = {}

    # Placeholder for calculations
    dataset = l1b_de_dict["imap_ultra_l1a_45sensor-de"]
    epoch = dataset.coords["epoch"].values

    pset_dict["epoch"] = epoch
    pset_dict["esa_step"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
