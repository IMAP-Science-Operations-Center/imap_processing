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
    filtered_dataset = extendedspin_dataset.sel(
        spin_number=extendedspin_dataset["spin_number"][good_mask]
    )

    cullingmask_dataset = create_dataset(filtered_dataset, name, "l1b", data_version)

    return cullingmask_dataset
