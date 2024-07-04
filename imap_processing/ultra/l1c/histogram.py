"""Calculate Histogram."""

import numpy as np
import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_histogram(histogram_dataset: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dictionary with defined datatype for Histogram Data.

    Parameters
    ----------
    histogram_dataset : xarray.Dataset
        Dataset containing histogram data.
    name : str
        Name of dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    histogram_dict = {}

    # Placeholder for calculations
    # TODO: come back and update this data structure.
    epoch = histogram_dataset.coords["epoch"].values

    histogram_dict["epoch"] = epoch
    histogram_dict["sid"] = np.zeros(len(epoch), dtype=np.uint8)

    dataset = create_dataset(histogram_dict, name, "l1c")

    return dataset
