"""Calculate Badtimes."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset, extract_data_dict

FILLVAL_UINT16 = 65535
FILLVAL_FLOAT32 = -1.0e31
FILLVAL_FLOAT64 = -1.0e31
FILLVAL_UINT32 = 4294967295


def calculate_badtimes(
    extendedspin_dataset: xr.Dataset,
    goodtimes_spins: NDArray,
    name: str,
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Badtimes Data.

    Parameters
    ----------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    goodtimes_spins : np.ndarray
        Dataset containing the culled data.
    name : str
        Name of the dataset.

    Returns
    -------
    badtimes_dataset : xarray.Dataset
        Dataset containing the extendedspin data that has been culled.
    """
    n_bins = extendedspin_dataset.sizes["energy_bin_geometric_mean"]
    culled_spins = np.setdiff1d(
        extendedspin_dataset["spin_number"].values, goodtimes_spins
    )
    filtered_dataset = extendedspin_dataset.sel(spin_number=culled_spins)

    data_dict = extract_data_dict(filtered_dataset)

    badtimes_dataset = create_dataset(data_dict, name, "l1b")

    if badtimes_dataset["spin_number"].size == 0:
        badtimes_dataset = badtimes_dataset.drop_dims("spin_number")
        badtimes_dataset = badtimes_dataset.expand_dims(spin_number=[FILLVAL_UINT32])
        badtimes_dataset["spin_start_time"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        badtimes_dataset["spin_period"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        badtimes_dataset["spin_rate"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        badtimes_dataset["start_pulses_per_spin"] = xr.DataArray(
            np.array([FILLVAL_FLOAT32], dtype="float32"),
            dims=["spin_number"],
        )
        badtimes_dataset["stop_pulses_per_spin"] = xr.DataArray(
            np.array([FILLVAL_FLOAT32], dtype="float32"),
            dims=["spin_number"],
        )
        badtimes_dataset["coin_pulses_per_spin"] = xr.DataArray(
            np.array([FILLVAL_FLOAT32], dtype="float32"),
            dims=["spin_number"],
        )
        badtimes_dataset["rejected_events_per_spin"] = xr.DataArray(
            np.array([FILLVAL_UINT32], dtype="uint32"),
            dims=["spin_number"],
        )
        badtimes_dataset["quality_attitude"] = xr.DataArray(
            np.array([FILLVAL_UINT16], dtype="uint16"), dims=["spin_number"]
        )
        badtimes_dataset["quality_hk"] = xr.DataArray(
            np.array([FILLVAL_UINT16], dtype="uint16"),
            dims=["spin_number"],
        )
        badtimes_dataset["quality_instruments"] = xr.DataArray(
            np.array([FILLVAL_UINT16], dtype="uint16"),
            dims=["spin_number"],
        )
        badtimes_dataset["quality_ena_rates"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((n_bins, 1), FILLVAL_UINT16, dtype="uint16"),
        )
        badtimes_dataset["ena_rates"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((n_bins, 1), FILLVAL_FLOAT64, dtype="float64"),
        )
        badtimes_dataset["ena_rates_threshold"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((n_bins, 1), FILLVAL_FLOAT32, dtype="float32"),
        )

    return badtimes_dataset
