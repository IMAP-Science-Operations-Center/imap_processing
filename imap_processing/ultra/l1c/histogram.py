"""Calculates Histogram."""

from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


def calculate_histogram(histogram_l1a_dataset, name, instrument_id):
    """
    Create dictionary with defined datatype for Histogram Data.

    Parameters
    ----------
    histogram_l1a_dataset: xarray.Dataset
        Dataset containing histogram data.
    name: str
        Name of the dataset.
    instrument_id: int
        Instrument ID.

    Returns
    -------
    histogram_l1c_dataset : xarray.Dataset
        Dataset containing the data.
    """
    keys_to_copy = [
        "epoch",
        "sid",
        "row",
        "column",
        "SHCOARSE",
        "SPIN",
        "PACKETDATA",
    ]

    histogram_dict = {key.lower(): histogram_l1a_dataset[key] for key in keys_to_copy}
    histogram_l1c_dataset = create_dataset(histogram_dict, name, "l1c", instrument_id)

    return histogram_l1c_dataset
