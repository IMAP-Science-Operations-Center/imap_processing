"""Calculate ULTRA L1c."""

import xarray as xr

from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1c.helio_pset import calculate_helio_pset
from imap_processing.ultra.l1c.spacecraft_pset import calculate_spacecraft_pset


def ultra_l1c(
    data_dict: dict, ancillary_files: dict, descriptor: str
) -> list[xr.Dataset]:
    """
    Will process ULTRA L1A and L1B data into L1C CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.
    ancillary_files : dict
        Ancillary files.
    descriptor : str
        Job descriptor.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
        List of xarray.Dataset.
    """
    output_datasets = []
    create_helio_pset = True if "helio" in descriptor else False
    # Account for the possibility of having 45 and 90 in the dictionary.
    for instrument_id in [45, 90]:
        if (
            f"imap_ultra_l1b_{instrument_id}sensor-goodtimes" in data_dict
            and f"imap_ultra_l1b_{instrument_id}sensor-de" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-rates" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-aux" in data_dict
            and create_helio_pset
        ):
            helio_pset = calculate_helio_pset(
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-de"],
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-goodtimes"],
                data_dict[f"imap_ultra_l1a_{instrument_id}sensor-rates"],
                data_dict[f"imap_ultra_l1a_{instrument_id}sensor-aux"],
                f"imap_ultra_l1c_{instrument_id}sensor-heliopset",
                ancillary_files,
                instrument_id,
                UltraConstants.TOFXPH_SPECIES_GROUPS["proton"],
            )
            output_datasets = [helio_pset]
        elif (
            f"imap_ultra_l1b_{instrument_id}sensor-goodtimes" in data_dict
            and f"imap_ultra_l1b_{instrument_id}sensor-de" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-rates" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-aux" in data_dict
        ):
            spacecraft_pset = calculate_spacecraft_pset(
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-de"],
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-goodtimes"],
                data_dict[f"imap_ultra_l1a_{instrument_id}sensor-rates"],
                data_dict[f"imap_ultra_l1a_{instrument_id}sensor-aux"],
                f"imap_ultra_l1c_{instrument_id}sensor-spacecraftpset",
                ancillary_files,
                instrument_id,
                UltraConstants.TOFXPH_SPECIES_GROUPS["proton"],
            )
            output_datasets = [spacecraft_pset]
            spacecraft_pset_non_proton = calculate_spacecraft_pset(
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-de"],
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-goodtimes"],
                data_dict[f"imap_ultra_l1a_{instrument_id}sensor-rates"],
                data_dict[f"imap_ultra_l1a_{instrument_id}sensor-aux"],
                f"imap_ultra_l1c_{instrument_id}sensor-spacecraftpset-nonproton",
                ancillary_files,
                instrument_id,
                UltraConstants.TOFXPH_SPECIES_GROUPS["non_proton"],
            )
            if spacecraft_pset_non_proton is not None:
                output_datasets.append(spacecraft_pset_non_proton)
    if not output_datasets:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
