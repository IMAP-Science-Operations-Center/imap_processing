"""Calculate Badtimes."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

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

    filtered_dataset = extendedspin_dataset.sel(spin_number=culled_spins)

    badtimes_dataset = create_dataset(filtered_dataset, name, "l1b")

    if badtimes_dataset["spin_number"].size == 0:
        badtimes_dataset = badtimes_dataset.drop_dims("spin_number")
        badtimes_dataset = badtimes_dataset.expand_dims(spin_number=[FILLVAL_UINT32])
        badtimes_dataset["spin_start_time"] = np.array(
            [FILLVAL_FLOAT64], dtype="float64"
        )
        badtimes_dataset["spin_period"] = np.array([FILLVAL_FLOAT64], dtype="float64")
        badtimes_dataset["spin_rate"] = np.array([FILLVAL_FLOAT64], dtype="float64")
        badtimes_dataset["quality_attitude"] = np.array(
            [FILLVAL_UINT16], dtype="uint16"
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
