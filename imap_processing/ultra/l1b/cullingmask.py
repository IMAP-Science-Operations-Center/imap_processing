"""Calculate Culling Mask."""

import xarray as xr

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_cullingmask(
    extendedspin_dataset: xr.Dataset, name: str, data_version: str
) -> xr.Dataset:
    """
    Create dataset with defined datatype for Culling Mask Data.

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
    cullingmask_dataset : xarray.Dataset
        Dataset containing the data.
    """
    cullingmask_dict = {}
    cullingmask_dict["spin_number"] = extendedspin_dataset["spin_number"]
    cullingmask_dict["median_rate_energy"] = extendedspin_dataset["median_rate_energy"]

    # TODO: add more data to cullingmask_dict.

    cullingmask_dataset = create_dataset(cullingmask_dict, name, "l1b", data_version)

    return cullingmask_dataset
