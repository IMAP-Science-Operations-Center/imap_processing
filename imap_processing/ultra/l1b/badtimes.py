"""Calculate Badtimes."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset, extract_data_dict

FILLVAL_UINT16 = 65535
FILLVAL_FLOAT64 = -1.0e31
FILLVAL_UINT32 = 4294967295


def calculate_badtimes(
    extendedspin_dataset: xr.Dataset,
    cullingmask_spins: NDArray,
    name: str,
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Badtimes Data.

    Parameters
    ----------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    cullingmask_spins : np.ndarray
        Dataset containing the culled data.
    name : str
        Name of the dataset.

    Returns
    -------
    badtimes_dataset : xarray.Dataset
        Dataset containing the extendedspin data that has been culled.
    """
    culled_spins = np.setdiff1d(
        extendedspin_dataset["spin_number"].values, cullingmask_spins
    )
    extendedspin_dataset = extendedspin_dataset.assign_coords(
        epoch=("spin_number", extendedspin_dataset["epoch"].values)
    )
    filtered_dataset = extendedspin_dataset.sel(spin_number=culled_spins)

    data_dict = extract_data_dict(filtered_dataset)

    badtimes_dataset = create_dataset(data_dict, name, "l1b")

    if badtimes_dataset["spin_number"].size == 0:
        badtimes_dataset = badtimes_dataset.drop_dims("spin_number")
        badtimes_dataset = badtimes_dataset.expand_dims(spin_number=[FILLVAL_UINT32])
        badtimes_dataset = badtimes_dataset.assign_coords(
            epoch=("spin_number", [extendedspin_dataset["epoch"].values[0]])
        )
        badtimes_dataset["spin_start_time"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        badtimes_dataset["spin_period"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        badtimes_dataset["spin_rate"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        badtimes_dataset["quality_attitude"] = xr.DataArray(
            np.array([FILLVAL_UINT16], dtype="uint16"), dims=["spin_number"]
        )
        badtimes_dataset["quality_ena_rates"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((3, 1), FILLVAL_UINT16, dtype="uint16"),
        )
        badtimes_dataset["ena_rates"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((3, 1), FILLVAL_FLOAT64, dtype="float64"),
        )

    return badtimes_dataset
