"""Calculate ULTRA L1c."""

import xarray as xr

from imap_processing.ultra.l1c.helio_pset import calculate_helio_pset
from imap_processing.ultra.l1c.histogram import calculate_histogram
from imap_processing.ultra.l1c.spacecraft_pset import calculate_spacecraft_pset


def ultra_l1c(
    data_dict: dict, ancillary_files: dict, has_spice: bool
) -> list[xr.Dataset]:
    """
    Will process ULTRA L1A and L1B data into L1C CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.
    ancillary_files : dict
        Ancillary files.
    has_spice : bool
        Whether to use SPICE data.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
        List of xarray.Dataset.
    """
    output_datasets = []

    # Account for possibility of having 45 and 90 in dictionary.
    for instrument_id in [45, 90]:
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
            and has_spice
        ):
            helio_pset = calculate_helio_pset(
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-de"],
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-extendedspin"],
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-cullingmask"],
                f"imap_ultra_l1c_{instrument_id}sensor-heliopset",
                ancillary_files,
            )
            output_datasets = [helio_pset]
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
                ancillary_files,
            )
            output_datasets = [spacecraft_pset]
    if not output_datasets:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
