"""Calculate Culling Mask."""

import xarray as xr

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags
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
    # If the spin rate was too high or low then the spin should be thrown out.
    valid_index = (
        extendedspin_dataset["quality_attitude"] & ImapAttitudeUltraFlags.SPINRATE.value
    ) == 0
    good_attitude_dataset = extendedspin_dataset.where(valid_index, drop=True)

    # If the rates at any energy level are too high then throw out the entire spin.
    high_rates_mask = (
        good_attitude_dataset["quality_ena_rates"] & ImapRatesUltraFlags.HIGHRATES.value
        == 0
    ).all(dim="energy_bin_geometric_mean")
    filtered_dataset = good_attitude_dataset.sel(
        spin_number=good_attitude_dataset["spin_number"][high_rates_mask]
    )
    dataset_dict = {
        **{var: filtered_dataset[var].values for var in filtered_dataset.data_vars},
        **{coord: filtered_dataset[coord].values for coord in filtered_dataset.coords},
    }

    cullingmask_dataset = create_dataset(dataset_dict, name, "l1b", data_version)

    return cullingmask_dataset
