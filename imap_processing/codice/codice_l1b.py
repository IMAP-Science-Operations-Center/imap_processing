"""
Perform CoDICE l1b processing.

This module processes CoDICE l1a files and creates L1b data products.

Notes
-----
from imap_processing.codice.codice_l1b import process_codice_l1b
dataset = process_codice_l1b(l1a_filenanme)
"""

import logging
from pathlib import Path

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_codice_l1b(file_path: Path) -> xr.Dataset:
    """
    Will process CoDICE l1a data to create l1b data products.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the CoDICE L1a file to process.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(f"\nProcessing {file_path}")

    # Open the l1a file
    l1a_dataset = load_cdf(file_path)

    # Use the logical source as a way to distinguish between data products and
    # set some useful distinguishing variables
    dataset_name = l1a_dataset.attrs["Logical_source"].replace("_l1a_", "_l1b_")
    descriptor = dataset_name.removeprefix("imap_codice_l1b_")
    apid = constants.CODICEAPID_MAPPING[descriptor]

    # Get the L1b CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1b")

    # Use the L1a data product as a starting point for L1b
    l1b_dataset = l1a_dataset.copy()

    # Update the global attributes
    l1b_dataset.attrs = cdf_attrs.get_global_attributes(dataset_name)

    # Determine which variables need to be converted from counts to rates
    # TODO: Figure out exactly which hskp variables need to be converted
    if descriptor == "hskp":
        data_variables = []
        support_variables = ["cmdexe", "cmdrjct"]
        variables_to_convert = support_variables
    else:
        data_variables = getattr(
            constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
        )
        support_variables = constants.DATA_PRODUCT_CONFIGURATIONS[apid][
            "support_variables"
        ]
        variables_to_convert = data_variables + support_variables

    for variable_name in variables_to_convert:
        # Apply conversion of data from counts to rates
        # TODO: Properly implement conversion factors on a per-data-product basis
        #       For now, just divide by 100 to get float values
        l1b_dataset[variable_name].data = l1b_dataset[variable_name].data / 100

        # Set the variable attributes
        if variable_name in data_variables:
            cdf_attrs_key = f"{descriptor}-{variable_name}"
        elif variable_name in support_variables:
            cdf_attrs_key = variable_name
        l1b_dataset[variable_name].attrs = cdf_attrs.get_variable_attributes(
            cdf_attrs_key
        )

    logger.info(f"\nFinal data product:\n{l1b_dataset}\n")

    return l1b_dataset
