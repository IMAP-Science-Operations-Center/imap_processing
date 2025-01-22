"""Calculate Extended Spin."""

import xarray as xr

from imap_processing.ultra.l1b.ultra_l1b_culling import flag_spin
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_extendedspin(
    rates_dataset: xr.Dataset, de_dataset: xr.Dataset, name: str
) -> xr.Dataset:
    """
    Create dataset with defined datatypes for Extended Spin Data.

    Parameters
    ----------
    rates_dataset : xarray.Dataset
        Dataset containing l1a rates data.
    de_dataset : xarray.Dataset
        Dataset containing l1b de data
    name : str
        Name of the dataset.

    Returns
    -------
    extendedspin_dataset : xarray.Dataset
        Dataset containing the data.
    """
    extendedspin_dict = {}
    quality_flags, spin, energy = flag_spin(
        de_dataset["event_times"],
        de_dataset["energy"],
    )

    # These will be the coordinates.
    extendedspin_dict["spin_number"] = spin
    extendedspin_dict["energy"] = energy

    # TODO
    # extendedspin_dict["ena_rates"]
    # extendedspin_dict["ena_rates_threshold"]
    # extendedspin_dict["spin_start_time"]
    # extendedspin_dict["avg_spin_period"]
    # extendedspin_dict["rate_start_pulses"]
    # extendedspin_dict["rate_stop_pulses"]
    # extendedspin_dict["rate_coin_pulses"]
    # extendedspin_dict["rate_processed_events"]
    # extendedspin_dict["rate_rejected_events"]
    # extendedspin_dict["quality_attitude"]
    # extendedspin_dict["quality_instruments"]

    extendedspin_dict["quality_ena_rates"] = quality_flags

    extendedspin_dataset = create_dataset(extendedspin_dict, name, "l1b")

    return extendedspin_dataset
