"""Calculate ULTRA L1c."""

import xarray as xr

from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import get_de_product_name
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

    # TODO
    # Determine which l1b priority DE product to use in creating the l1c products.
    # This will vary per-pointing by an ancillary file produced by the ULTRA team.

    # Account for the possibility of having 45 and 90 in the dictionary.
    for instrument_id in [45, 90]:
        # All l1c products require a l1b de dependency so check that first
        # and calculate the correct l1b de product to use based on the repointing ID
        # and ancillary files.
        if f"imap_ultra_l1b_{instrument_id}sensor-de" in data_dict:
            # get repoint number
            repoint = data_dict[f"imap_ultra_l1b_{instrument_id}sensor-de"].attrs.get(
                "Repointing", None
            )
            if repoint is None:
                raise ValueError("Repointing ID attribute is missing from the dataset.")
            # Determine which l1b de product to use in calculating the l1c products.
            # Will be either the raw de product or a priority 1-4 de product.
            de_product_desc = get_de_product_name(
                repoint, instrument_id, "l1c", ancillary_files
            )
            if de_product_desc not in data_dict:
                raise ValueError(
                    f"Selected L1B DE product '{de_product_desc}' for instrument "
                    f"{instrument_id} is not present in data_dict. Available L1B DE "
                    f"products: {data_dict.keys()}"
                )
        else:
            continue
        if (
            f"imap_ultra_l1b_{instrument_id}sensor-goodtimes" in data_dict
            and de_product_desc in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-rates" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-aux" in data_dict
            and create_helio_pset
        ):
            helio_pset = calculate_helio_pset(
                data_dict[de_product_desc],
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
            and de_product_desc in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-rates" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-aux" in data_dict
        ):
            spacecraft_pset = calculate_spacecraft_pset(
                data_dict[de_product_desc],
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
                data_dict[de_product_desc],
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
