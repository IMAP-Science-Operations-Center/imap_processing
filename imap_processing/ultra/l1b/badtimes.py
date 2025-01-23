"""Calculate Badtimes."""

import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_badtimes(
    extendedspin_dataset: xr.Dataset, name: str, data_version: str
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Badtimes Data.

    Parameters
    ----------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    name : str
        Name of the dataset.
    data_version : str
        Version of the data.

    Returns
    -------
    badtimes_dataset : xarray.Dataset
        Dataset containing the data.
    """
    badtimes_dict = {}
    badtimes_dict["spin_number"] = extendedspin_dataset["spin_number"]
    badtimes_dict["median_rate_energy"] = extendedspin_dataset["median_rate_energy"]

    # TODO: add more data to badtimes_dict.

    badtimes_dataset = create_dataset(badtimes_dict, name, "l1b", data_version)

    return badtimes_dataset
