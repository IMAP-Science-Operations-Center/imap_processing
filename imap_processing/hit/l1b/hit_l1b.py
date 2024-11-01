"""IMAP-HIT L1B data processing."""

import logging

import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.hit_utils import (
    HitAPID,
    get_attribute_manager,
    process_housekeeping,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l1b(dependencies: dict, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT data to L1B.

    Parameters
    ----------
    dependencies : dict
        Dictionary of dependencies that are L1A xarray datasets
        for science data and a file path string to a CCSDS file
        for housekeeping data.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    cdf_filepaths : xarray.Dataset
        L1B processed data.
    """
    # Create datasets
    datasets = []
    if "imap_hit_l0_raw" in dependencies:
        packet_file = dependencies["imap_hit_l0_raw"]
        datasets = create_l1b_hk_dataset(packet_file, data_version)
    elif "imap_hit_l1a_countrates" in dependencies:
        # TODO: process science data. placeholder for future code
        pass

    return datasets


def create_l1b_hk_dataset(packet_file: str, data_version: str) -> list[xr.Dataset]:
    """
    Create HIT L1B housekeeping dataset.

    Reads in L0 CCSDS file using packet_file_to_datasets with
    use_derived_value=True to get housekeeping data with
    engineering units as a Xarray Dataset. This dataset is then
    updated to replace the leak variables with a single
    leak_i 2D array and add the dataset attributes and
    coordinates, and data variable dimensions according to
    specifications in a cdf yaml file.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    dataset : xarray.Dataset
        An updated dataset ready for CDF conversion.
    """
    logger.info("Creating HIT L1B housekeeping dataset")

    # Unpack ccsds file
    packet_definition = (
        imap_module_directory / "hit/packet_definitions/hit_packet_definitions.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file,
        xtce_packet_definition=packet_definition,
        use_derived_value=True,
    )

    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager(data_version, "l1b")

    # Process housekeeping to l1b.
    logical_source = "imap_hit_l1b_hk"
    dataset = process_housekeeping(
        datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, logical_source
    )
    logger.info("HIT L1B housekeeping dataset created")
    return [dataset]
