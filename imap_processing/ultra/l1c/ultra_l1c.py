"""Calculates ULTRA L1c."""

from imap_processing.ultra.l1c.histogram import calculate_histogram
from imap_processing.ultra.l1c.pset import (
    calculate_pset_backgroundrates,
    calculate_pset_counts,
    calculate_pset_exposure,
    calculate_pset_sensitivity,
)


def ultra_l1c(data_dict: dict):
    """
    Process ULTRA L1A and L1B data into L1C CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.

    Returns
    -------
    output_datasets : list of xarray.Dataset
        List of xarray.Dataset
    """
    output_datasets = []
    instrument_id = 45 if any("45" in key for key in data_dict.keys()) else 90

    if (
        f"imap_ultra_l1a_{instrument_id}sensor-histogram" in data_dict
        and f"imap_ultra_l1b_{instrument_id}sensor-cullingmask" in data_dict
    ):
        histogram_dataset = calculate_histogram(
            data_dict[f"imap_ultra_l1a_{instrument_id}sensor-histogram"],
            data_dict[f"imap_ultra_l1b_{instrument_id}sensor-cullingmask"],
            f"imap_ultra_l1c_{instrument_id}sensor-histogram",
        )
        output_datasets.append(histogram_dataset)
    elif (
        f"imap_ultra_l1b_{instrument_id}sensor-cullingmask" in data_dict
        and f"imap_ultra_l1b_{instrument_id}sensor-de" in data_dict
        and f"imap_ultra_l1b_{instrument_id}extendedspin-de" in data_dict
    ):
        pset_exposure_dataset = calculate_pset_exposure(
            data_dict, f"imap_ultra_l1c_{instrument_id}sensor-pset-exposure"
        )
        pset_sensitivity_dataset = calculate_pset_sensitivity(
            data_dict, f"imap_ultra_l1c_{instrument_id}sensor-pset-sensitivity"
        )
        pset_backgroundrates_dataset = calculate_pset_backgroundrates(
            data_dict, f"imap_ultra_l1c_{instrument_id}sensor-pset-backgroundrates"
        )
        pset_counts_dataset = calculate_pset_counts(
            data_dict, f"imap_ultra_l1c_{instrument_id}sensor-pset-counts"
        )

        output_datasets.extend(
            [
                pset_exposure_dataset,
                pset_sensitivity_dataset,
                pset_backgroundrates_dataset,
                pset_counts_dataset,
            ]
        )
    else:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
