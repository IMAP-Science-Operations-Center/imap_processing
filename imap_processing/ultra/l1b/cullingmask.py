"""Calculates Culling Mask."""

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_cullingmask(extendedspin_dataset, name):
    """
    Create dataset with defined datatype for Culling Mask Data.

    Parameters
    ----------
    extendedspin_dataset: xarray.Dataset
        Dataset containing rates data.
    name: str
        Name of the dataset.

    Returns
    -------
    cullingmask_dataset : xarray.Dataset
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

    cullingmask_dict = {key: extendedspin_dataset[key] for key in keys_to_copy}
    cullingmask_dataset = create_dataset(cullingmask_dict, name, "l1b")

    return cullingmask_dataset
