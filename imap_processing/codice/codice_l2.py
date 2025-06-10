"""
Perform CoDICE l2 processing.

This module processes CoDICE l1 files and creates L2 data products.

Notes
-----
from imap_processing.codice.codice_l2 import process_codice_l2
dataset = process_codice_l2(l1_filename)
"""

import logging
from pathlib import Path

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_codice_l2(file_path: Path) -> xr.Dataset:
    """
    Will process CoDICE l1 data to create l2 data products.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the CoDICE L1 file to process.

    Returns
    -------
    l2_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(f"Processing {file_path}")

    # Open the l1 file
    l1_dataset = load_cdf(file_path)

    # Use the logical source as a way to distinguish between data products and
    # set some useful distinguishing variables
    # TODO: Could clean this up by using imap-data-access methods?
    dataset_name = l1_dataset.attrs["Logical_source"]
    data_level = dataset_name.removeprefix("imap_codice_").split("_")[0]
    descriptor = dataset_name.removeprefix(f"imap_codice_{data_level}_")
    dataset_name = dataset_name.replace(data_level, "l2")

    # TODO: Temporary work-around to replace "PHA" naming convention with
    #       "direct events" This will eventually be changed at the L1 level and
    #       thus this will eventually be removed.
    if descriptor == "lo-pha":
        dataset_name = dataset_name.replace("lo-pha", "lo-direct-events")
    elif descriptor == "hi-pha":
        dataset_name = dataset_name.replace("hi-pha", "hi-direct-events")

    # Use the L1 data product as a starting point for L2
    l2_dataset = l1_dataset.copy()

    # Get the L2 CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l2")

    # Update the global attributes
    l2_dataset.attrs = cdf_attrs.get_global_attributes(dataset_name)

    # Set the variable attributes
    for variable_name in l2_dataset:
        l2_dataset[variable_name].attrs = cdf_attrs.get_variable_attributes(
            variable_name, check_schema=False
        )

    # TODO: Add L2-specific algorithms/functionality here. For SIT-4, we can
    #       just keep the data as-is.

    logger.info(f"\nFinal data product:\n{l2_dataset}\n")

    return l2_dataset
