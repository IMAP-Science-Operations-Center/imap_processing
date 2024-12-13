"""Decommutate HIT CCSDS data and create L1a data products."""

import logging

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
)
from imap_processing.hit.l0.decom_hit import decom_hit

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


def hit_l1a(packet_file: str, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT L0 data into L1A data products.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of Datasets of L1A processed data.
    """
    # Unpack ccsds file to xarray datasets
    datasets_by_apid = get_datasets_by_apid(packet_file)

    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager(data_version, "l1a")

    # Process l1a data products
    if HitAPID.HIT_HSKP in datasets_by_apid:
        logger.info("Creating HIT L1A housekeeping dataset")
        datasets_by_apid[HitAPID.HIT_HSKP] = process_housekeeping_data(
            datasets_by_apid[HitAPID.HIT_HSKP], attr_mgr, "imap_hit_l1a_hk"
        )

    if HitAPID.HIT_SCIENCE in datasets_by_apid:
        # TODO complete science data processing
        print("Skipping science data for now")
        datasets_by_apid[HitAPID.HIT_SCIENCE] = process_science(
            datasets_by_apid[HitAPID.HIT_SCIENCE], attr_mgr
        )

    return list(datasets_by_apid.values())


def process_science(dataset: xr.Dataset, attr_mgr: ImapCdfAttributes) -> xr.Dataset:
    """
    Will process science dataset for CDF product.

    Process binary science data for CDF creation. The data is
    grouped into science frames, decommutated and decompressed,
    and split into count rates and event datasets. Updates the
    dataset attributes and coordinates and data variable
    dimensions according to specifications in a cdf yaml file.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing HIT science data.

    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.

    Returns
    -------
    dataset : xarray.Dataset
        An updated dataset ready for CDF conversion.
    """
    logger.info("Creating HIT L1A science datasets")

    # Logical sources for the two products.
    # logical_sources = ["imap_hit_l1a_count-rates", "imap_hit_l1a_pulse-height-event"]

    # Decommutate and decompress the science data
    sci_dataset = decom_hit(dataset)

    # TODO: Complete this function
    #  - split the science data into count rates and event datasets
    #  - update dimensions and add attributes to the dataset and data arrays
    #  - return list of two datasets (count rates and events)?

    # logger.info("HIT L1A event dataset created")
    # logger.info("HIT L1A count rates dataset created")

    return sci_dataset
