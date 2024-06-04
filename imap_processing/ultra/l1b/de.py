"""Calculates Annotated Direct Events."""

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
    dataset = create_dataset(data_dict, name, "l1b")

    return dataset
