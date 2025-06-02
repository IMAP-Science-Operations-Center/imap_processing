"""
Perform CoDICE l1b processing.

This module processes CoDICE l1a files and creates L1b data products.

Notes
-----
from imap_processing.codice.codice_l1b import process_codice_l1b
dataset = process_codice_l1b(l1a_filenanme)
"""

# TODO: Figure out how to convert hi-priority data product. Need an updated
#       algorithm document that describes this.

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_to_rates(
    dataset: xr.Dataset, descriptor: str, variable_name: str
) -> np.ndarray:
    """
    Apply a conversion from counts to rates.

    The formula for conversion from counts to rates is specific to each data
    product, but is largely grouped by CoDICE-Lo and CoDICE-Hi products.

    Parameters
    ----------
    dataset : xr.Dataset
        The L1b dataset containing the data to convert.
    descriptor : str
        The descriptor of the data product of interest.
    variable_name : str
        The variable name to apply the conversion to.

    Returns
    -------
    rates_data : np.ndarray
        The converted data array.
    """
    acq_times = dataset.epoch.data
    acq_times = 1

    if descriptor in [
        "lo-counters-aggregated",
        "lo-counters-singles",
        "lo-nsw-angular",
        "lo-sw-angular",
        "lo-nsw-priority",
        "lo-sw-priority",
        "lo-nsw-species",
        "lo-sw-species",
        "lo-ialirt",
    ]:
        rates_data = dataset[variable_name].data / (
            acq_times
            * 10e-6
            * constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spin_sectors"]
        )
    elif descriptor in [
        "hi-counters-aggregated",
        "hi-counters-singles",
        "hi-omni",
        "hi-priority",
        "hi-sectored",
        "hi-ialirt",
    ]:
        rates_data = dataset[variable_name].data / (
            constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spin_sectors"]
            * constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spins"]
            * acq_times
        )
    elif descriptor == "hskp":
        rates_data = dataset[variable_name].data / acq_times

    return rates_data


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

    # Direct event data products do not have a level L1B
    if descriptor in ["lo-pha", "hi-pha"]:
        logger.warning("Encountered direct event data product. Skipping L1b processing")
        return None

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
    # Housekeeping and binned datasets are treated a bit differently since
    # not all variables need to be converted
    if descriptor == "hskp":
        data_variables = []
        support_variables = ["cmdexe", "cmdrjct"]
    elif descriptor == "hi-sectored":
        data_variables = ["h", "he3he4", "cno", "fe"]
        support_variables = []
    elif descriptor == "hi-omni":
        data_variables = ["h", "he3", "he4", "c", "o", "ne_mg_si", "fe", "uh"]
        support_variables = []
    elif descriptor == "hi-ialirt":
        data_variables = ["h"]
        support_variables = []
    else:
        data_variables = getattr(
            constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
        )
        support_variables = constants.DATA_PRODUCT_CONFIGURATIONS[apid][
            "support_variables"
        ]
    variables_to_convert = data_variables + support_variables

    for variable_name in variables_to_convert:
        l1b_dataset[variable_name].data = convert_to_rates(
            l1b_dataset, descriptor, variable_name
        )

        # Set the variable attributes
        if variable_name in data_variables:
            cdf_attrs_key = f"{descriptor}-{variable_name}"
        elif variable_name in support_variables:
            cdf_attrs_key = variable_name
        l1b_dataset[variable_name].attrs = cdf_attrs.get_variable_attributes(
            cdf_attrs_key, check_schema=False
        )

    # TODO: Temporary workaround to avoid xarray_to_cdf error
    l1b_dataset.epoch.attrs["DEPEND_0"] = " "

    logger.info(f"\nFinal data product:\n{l1b_dataset}\n")

    return l1b_dataset
