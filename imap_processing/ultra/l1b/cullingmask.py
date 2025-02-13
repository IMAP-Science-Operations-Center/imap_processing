"""Calculate Culling Mask."""

import xarray as xr

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags


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
    rate_mask = (
        extendedspin_dataset["quality_attitude"] & ImapAttitudeUltraFlags.SPINRATE.value
    ) == 0
    good_attitude = extendedspin_dataset["spin_number"].values[rate_mask]
    good_attitude_dataset = extendedspin_dataset.sel(spin_number=good_attitude)

    # Counts in every energy bin should be equal to zero to throw out the spin.
    zero_counts = (
        (
            good_attitude_dataset["quality_ena_rates"]
            & ImapRatesUltraFlags.ZEROCOUNTS.value
        )
        != 0
    ).all(dim="energy_bin_geometric_mean")
    zero_counts_mask = ~zero_counts
    non_zero_dataset = good_attitude_dataset.sel(
        spin_number=good_attitude_dataset["spin_number"][zero_counts_mask]
    )

    # If the rates at any energy level are too high then throw out the entire spin.
    high_rates_mask = (
        non_zero_dataset["quality_ena_rates"] & ImapRatesUltraFlags.HIGHRATES.value == 0
    ).all(dim="energy_bin_geometric_mean")
    cullingmask_dataset = non_zero_dataset.sel(
        spin_number=non_zero_dataset["spin_number"][high_rates_mask]
    )

    return cullingmask_dataset
