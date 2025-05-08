"""Calculate ULTRA L1c."""

import xarray as xr

from imap_processing.ultra.l1c.histogram import calculate_histogram
from imap_processing.ultra.l1c.spacecraft_pset import calculate_spacecraft_pset


def ultra_l1c(data_dict: dict) -> list[xr.Dataset]:
    """
    Will process ULTRA L1A and L1B data into L1C CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
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
        spacecraft_pset = calculate_spacecraft_pset(
            data_dict[f"imap_ultra_l1b_{instrument_id}sensor-de"],
            data_dict[f"imap_ultra_l1b_{instrument_id}sensor-extendedspin"],
            data_dict[f"imap_ultra_l1b_{instrument_id}sensor-cullingmask"],
            f"imap_ultra_l1c_{instrument_id}sensor-spacecraftpset",
        )
        # TODO: add calculate_helio_pset here
        output_datasets = [spacecraft_pset]
    else:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
