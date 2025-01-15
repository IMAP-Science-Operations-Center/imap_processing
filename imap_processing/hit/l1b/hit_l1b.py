"""IMAP-HIT L1B data processing."""

import logging

import xarray as xr

from imap_processing.hit.hit_utils import (
    HitAPID,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
)

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l1b(dependencies: dict, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT data to L1B.

    Processes dependencies needed to create L1B data products.

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
    processed_data : list[xarray.Dataset]
        List of L1B datasets.
    """
    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager(data_version, "l1b")

    # Create L1B datasets
    datasets: list = []
    if "imap_hit_l0_raw" in dependencies:
        # Unpack ccsds file to xarray datasets
        packet_file = dependencies["imap_hit_l0_raw"]
        datasets_by_apid = get_datasets_by_apid(packet_file, derived=True)
        # Process housekeeping to l1b.
        datasets.append(
            process_housekeeping_data(
                datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1b_hk"
            )
        )
        logger.info("HIT L1B housekeeping dataset created")
    if "imap_hit_l1a_countrates" in dependencies:
        # TODO: process science data. placeholder for future code
        pass

    return datasets
