"""Calculate Goodtimes."""

import numpy as np
import xarray as xr

from imap_processing.ultra.l1b.quality_flag_filters import SPIN_QUALITY_FLAG_FILTERS
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset, extract_data_dict

FILLVAL_UINT16 = 65535
FILLVAL_FLOAT32 = -1.0e31
FILLVAL_FLOAT64 = -1.0e31
FILLVAL_UINT32 = 4294967295


def calculate_goodtimes(extendedspin_dataset: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dataset with defined datatype for Goodtimes Data.

    Parameters
    ----------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    name : str
        Name of the dataset.

    Returns
    -------
    goodtimes_dataset : xarray.Dataset
        Dataset containing the extendedspin data that remains after culling.
    """
    n_bins = extendedspin_dataset.sizes["energy_bin_geometric_mean"]
    # If the spin rate was too high or low then the spin should be thrown out.
    # If the rates at any energy level are too high then throw out the entire spin.
    good_mask = (
        (
            extendedspin_dataset["quality_attitude"]
            & sum(flag.value for flag in SPIN_QUALITY_FLAG_FILTERS["quality_attitude"])
        )
        == 0
    ) & (
        (
            (
                extendedspin_dataset["quality_ena_rates"]
                & sum(
                    flag.value
                    for flag in SPIN_QUALITY_FLAG_FILTERS["quality_ena_rates"]
                )
            )
            == 0
        ).all(dim="energy_bin_geometric_mean")
    )
    filtered_dataset = extendedspin_dataset.sel(
        spin_number=extendedspin_dataset["spin_number"][good_mask]
    )

    data_dict = extract_data_dict(filtered_dataset)

    goodtimes_dataset = create_dataset(data_dict, name, "l1b")

    if goodtimes_dataset["spin_number"].size == 0:
        goodtimes_dataset = goodtimes_dataset.drop_dims("spin_number")
        goodtimes_dataset = goodtimes_dataset.expand_dims(spin_number=[FILLVAL_UINT32])
        goodtimes_dataset["spin_start_time"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        goodtimes_dataset["spin_period"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        goodtimes_dataset["spin_rate"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        goodtimes_dataset["start_pulses_per_spin"] = xr.DataArray(
            np.array([FILLVAL_FLOAT32], dtype="float32"),
            dims=["spin_number"],
        )
        goodtimes_dataset["stop_pulses_per_spin"] = xr.DataArray(
            np.array([FILLVAL_FLOAT32], dtype="float32"),
            dims=["spin_number"],
        )
        goodtimes_dataset["coin_pulses_per_spin"] = xr.DataArray(
            np.array([FILLVAL_FLOAT32], dtype="float32"),
            dims=["spin_number"],
        )
        goodtimes_dataset["rejected_events_per_spin"] = xr.DataArray(
            np.array([FILLVAL_UINT32], dtype="uint32"),
            dims=["spin_number"],
        )
        goodtimes_dataset["quality_attitude"] = xr.DataArray(
            np.array([FILLVAL_UINT16], dtype="uint16"), dims=["spin_number"]
        )
        goodtimes_dataset["quality_hk"] = xr.DataArray(
            np.array([FILLVAL_UINT16], dtype="uint16"),
            dims=["spin_number"],
        )
        goodtimes_dataset["quality_instruments"] = xr.DataArray(
            np.array([FILLVAL_UINT16], dtype="uint16"),
            dims=["spin_number"],
        )
        goodtimes_dataset["quality_ena_rates"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((n_bins, 1), FILLVAL_UINT16, dtype="uint16"),
        )
        goodtimes_dataset["ena_rates"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((n_bins, 1), FILLVAL_FLOAT64, dtype="float64"),
        )
        goodtimes_dataset["ena_rates_threshold"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((n_bins, 1), FILLVAL_FLOAT32, dtype="float32"),
        )

    return goodtimes_dataset
