"""Calculate Badtimes."""

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_badtimes(extendedspin_dataset, name):
    """
    Create dataset with defined datatypes for Badtimes Data.

    Parameters
    ----------
    extendedspin_dataset: xarray.Dataset
        Dataset containing rates data.
    name: str
        Name of the dataset.

    Returns
    -------
    badtimes_dataset : xarray.Dataset
        Dataset containing the data.
    """
    keys_to_copy = [
        "epoch",
        "spin_number",
        "spin_start_time",
        "avg_spin_period",
        "rate_start_pulses",
        "rate_stop_pulses",
        "rate_coin_pulses",
        "rate_processed_events",
        "rate_rejected_events",
    ]

    badtimes_dict = {key: extendedspin_dataset[key] for key in keys_to_copy}
    badtimes_dataset = create_dataset(badtimes_dict, name, "l1b")

    return badtimes_dataset
