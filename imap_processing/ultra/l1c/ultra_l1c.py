"""Calculate ULTRA L1c."""

from imap_processing.ultra.l1c.histogram import calculate_histogram
from imap_processing.ultra.l1c.pset import calculate_pset


def ultra_l1c(data_dict: dict):
    """
    Will process ULTRA L1A and L1B data into L1C CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.

    Returns
    -------
    list
        List of xarray.Dataset.
    """
    instrument_id = 45 if any("45" in key for key in data_dict.keys()) else 90

    if (
        f"imap_ultra_l1a_{instrument_id}sensor-histogram" in data_dict
        and f"imap_ultra_l1b_{instrument_id}sensor-cullingmask" in data_dict
    ):
        histogram_dataset = calculate_histogram(
            data_dict[f"imap_ultra_l1a_{instrument_id}sensor-histogram"],
            f"imap_ultra_l1c_{instrument_id}sensor-histogram",
        )
        output_datasets = [histogram_dataset]
    elif (
        f"imap_ultra_l1b_{instrument_id}sensor-cullingmask" in data_dict
        and f"imap_ultra_l1b_{instrument_id}sensor-de" in data_dict
        and f"imap_ultra_l1b_{instrument_id}sensor-extendedspin" in data_dict
    ):
        pset_dataset = calculate_pset(
            data_dict[f"imap_ultra_l1b_{instrument_id}sensor-de"],
            f"imap_ultra_l1c_{instrument_id}sensor-pset",
        )

        output_datasets = [pset_dataset]
    else:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
