"""Calculate Culling Mask."""

import numpy as np
import xarray as xr

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset, extract_data_dict

FILLVAL_UINT16 = 65535
FILLVAL_FLOAT64 = -1.0e31
FILLVAL_UINT32 = 4294967295


def calculate_cullingmask(extendedspin_dataset: xr.Dataset, name: str) -> xr.Dataset:
    """
    Create dataset with defined datatype for Culling Mask Data.

    Parameters
    ----------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    name : str
        Name of the dataset.

    Returns
    -------
    cullingmask_dataset : xarray.Dataset
        Dataset containing the extendedspin data that remains after culling.
    """
    # If the spin rate was too high or low then the spin should be thrown out.
    # If the rates at any energy level are too high then throw out the entire spin.
    good_mask = (
        (
            extendedspin_dataset["quality_attitude"]
            & ImapAttitudeUltraFlags.SPINRATE.value
        )
        == 0
    ) & (
        (
            (
                extendedspin_dataset["quality_ena_rates"]
                & ImapRatesUltraFlags.HIGHRATES.value
            )
            == 0
        ).all(dim="energy_bin_geometric_mean")
    )
    extendedspin_dataset = extendedspin_dataset.assign_coords(
        epoch=("spin_number", extendedspin_dataset["epoch"].values)
    )
    filtered_dataset = extendedspin_dataset.sel(
        spin_number=extendedspin_dataset["spin_number"][good_mask]
    )

    data_dict = extract_data_dict(filtered_dataset)

    cullingmask_dataset = create_dataset(data_dict, name, "l1b")

    if cullingmask_dataset["spin_number"].size == 0:
        cullingmask_dataset = cullingmask_dataset.drop_dims("spin_number")
        cullingmask_dataset = cullingmask_dataset.expand_dims(
            spin_number=[FILLVAL_UINT32]
        )
        cullingmask_dataset = cullingmask_dataset.assign_coords(
            epoch=("spin_number", [extendedspin_dataset["epoch"].values[0]])
        )
        cullingmask_dataset["spin_start_time"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        cullingmask_dataset["spin_period"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        cullingmask_dataset["spin_rate"] = xr.DataArray(
            np.array([FILLVAL_FLOAT64], dtype="float64"), dims=["spin_number"]
        )
        cullingmask_dataset["quality_attitude"] = xr.DataArray(
            np.array([FILLVAL_UINT16], dtype="uint16"), dims=["spin_number"]
        )
        cullingmask_dataset["quality_ena_rates"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((3, 1), FILLVAL_UINT16, dtype="uint16"),
        )
        cullingmask_dataset["ena_rates"] = (
            ("energy_bin_geometric_mean", "spin_number"),
            np.full((3, 1), FILLVAL_FLOAT64, dtype="float64"),
        )

    return cullingmask_dataset
