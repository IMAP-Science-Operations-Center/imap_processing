"""Calculate Annotated Direct Events."""

import numpy as np

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_de(de_dataset, name):
    """
    Create dataset with defined datatypes for Direct Event Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Dataset containing direct event data.
    name : str
        Name of the dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    de_dict = {}

    # Placeholder for calculations
    epoch = de_dataset.coords["epoch"].values

    de_dict["epoch"] = epoch
    de_dict["x_front"] = np.zeros(len(epoch), dtype=np.uint64)

    dataset = create_dataset(de_dict, name, "l1b")

    return dataset
