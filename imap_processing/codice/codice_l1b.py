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

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice import constants
from imap_processing.codice.utils import CODICEAPID
from imap_processing.utils import packet_file_to_datasets

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
    dataset : xarray.Dataset
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
    if descriptor in [
        "lo-counters-aggregated",
        "lo-counters-singles",
        "lo-nsw-angular",
        "lo-sw-angular",
        "lo-nsw-priority",
        "lo-sw-priority",
        "lo-ialirt",
    ]:
        # Applying rate calculation described in section 10.2 of the algorithm
        # document
        # In order to divide by acquisition times, we must reshape the acq
        # time data array to match the data variable shape
        dims = [1] * dataset[variable_name].data.ndim
        dims[1] = 128
        acq_times = dataset.acquisition_time_per_step.data.reshape(dims)  # (128)
        # Now perform the calculation
        rates_data = dataset[variable_name].data / (
            acq_times
            * 1e-3  # Converting from milliseconds to seconds
            * constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spin_sectors"]
        )
    elif descriptor in [
        "lo-nsw-species",
        "lo-sw-species",
    ]:
        # Applying rate calculation described in section 10.2 of the algorithm
        # document
        # In order to divide by acquisition times, we must reshape the acq
        # time data array to match the data variable shape (epoch, esa_step, sector)
        dims = [1] * dataset[variable_name].data.ndim
        dims[1] = 128
        acq_times = dataset.acquisition_time_per_step.data.reshape(dims)  # (128)
        # acquisition time have an array of shape (128,). We match n_sector to that.
        # Per CoDICE, fill first 127 with default value of 12. Then fill last with 11.
        n_sector = np.full(128, 12, dtype=int)
        n_sector[-1] = 11

        # Now perform the calculation
        rates_data = dataset[variable_name].data / (
            acq_times
            * 1e-3  # Converting from milliseconds to seconds
            * n_sector[:, np.newaxis]  # Spin sectors
        )
    elif descriptor in [
        "hi-counters-aggregated",
        "hi-counters-singles",
        "hi-omni",
        "hi-priority",
        "hi-sectored",
        "hi-ialirt",
    ]:
        # Applying rate calculation described in section 10.1 of the algorithm
        # document
        rates_data = dataset[variable_name].data / (
            constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spin_sectors"]
            * constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spins"]
            * constants.HI_ACQUISITION_TIME
        )

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

    # Direct event data products do not have a level L1B
    if descriptor in ["lo-direct-events", "hi-direct-events"]:
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

    # TODO: This was thrown together quickly and should be double-checked
    if descriptor == "hskp":
        xtce_filename = "codice_packet_definition.xml"
        xtce_packet_definition = Path(
            f"{imap_module_directory}/codice/packet_definitions/{xtce_filename}"
        )
        packet_file = (
            imap_module_directory
            / "tests"
            / "codice"
            / "data"
            / "imap_codice_l0_raw_20241110_v001.pkts"
        )
        datasets: dict[int, xr.Dataset] = packet_file_to_datasets(
            packet_file, xtce_packet_definition, use_derived_value=True
        )
        l1b_dataset = datasets[CODICEAPID.COD_NHK]

        # TODO: Drop the same variables as we do in L1a? (see line 1103 in
        #       codice_l1a.py

    else:
        variables_to_convert = getattr(
            constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
        )

        # Apply the conversion to rates
        for variable_name in variables_to_convert:
            l1b_dataset[variable_name].data = convert_to_rates(
                l1b_dataset, descriptor, variable_name
            )
            # Set the variable attributes
            cdf_attrs_key = f"{descriptor}-{variable_name}"
            l1b_dataset[variable_name].attrs = cdf_attrs.get_variable_attributes(
                cdf_attrs_key, check_schema=False
            )

        if descriptor in ["lo-sw-species", "lo-nsw-species"]:
            # Do not carry these variable attributes from L1a to L1b
            drop_variables = [
                "k_factor",
                "nso_half_spin",
                "sw_bias_gain_mode",
                "st_bias_gain_mode",
                "spin_period",
            ]
            l1b_dataset = l1b_dataset.drop_vars(drop_variables)

    logger.info(f"\nFinal data product:\n{l1b_dataset}\n")

    return l1b_dataset
