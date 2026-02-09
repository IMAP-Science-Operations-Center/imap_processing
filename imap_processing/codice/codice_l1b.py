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

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice import constants

logger = logging.getLogger(__name__)


def convert_to_rates(dataset: xr.Dataset, descriptor: str) -> np.ndarray:
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

    Returns
    -------
    rates_data : np.ndarray
        The converted data array.
    """
    # No uncertainty calculation for diagnostic counters products
    calculate_unc = False if "counters" in descriptor else True
    # Variables to convert based on descriptor
    variables_to_convert = getattr(
        constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
    )

    if descriptor.startswith("lo-"):
        # Calculate energy_table using voltage_table and k_factor
        energy_attrs = dataset["voltage_table"].attrs | {
            "UNITS": "keV/e",
            "LABLAXIS": "E/q",
            "CATDESC": "Energy per charge",
            "FIELDNAM": "Energy per charge",
        }
        # 1e3 is to convert eV to keV
        dataset["energy_table"] = xr.DataArray(
            dataset["voltage_table"].values * dataset["k_factor"].values * 1e-3,
            dims=[
                "esa_step",
            ],
            attrs=energy_attrs,
        )

    if descriptor in [
        "lo-counters-aggregated",
        "lo-counters-singles",
        "lo-nsw-angular",
        "lo-sw-angular",
        "lo-nsw-priority",
        "lo-sw-priority",
    ]:
        # Denominator to convert counts to rates
        denominator = (
            dataset.acquisition_time_per_esa_step
            * constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spin_sectors"]
        )

        # Do not carry these variable attributes from L1a to L1b for above products
        drop_variables = [
            "k_factor",
            "nso_half_spin",
            "sw_bias_gain_mode",
            "st_bias_gain_mode",
            "spin_period",
            "voltage_table",
            # TODO: undo this when I get new validation file from Joey
            # "acquisition_time_per_esa_step",
        ]
        dataset = dataset.drop_vars(drop_variables)
    elif descriptor in [
        "lo-nsw-species",
        "lo-sw-species",
        "lo-ialirt",
    ]:
        # Create n_sector with 'epoch' and 'esa_step' dimension. This is done by
        # xr.full_like with input dataset.acquisition_time_per_esa_step. This ensures
        # that the resulting n_sector has the same dimensions as
        # acquisition_time_per_esa_step. Per CoDICE, fill first 127 with default value
        # of 12. Then fill last with 11. In your SDC processing
        n_sector = xr.full_like(
            dataset.acquisition_time_per_esa_step, 12.0, dtype=np.float64
        )
        n_sector[:, -1] = 11.0

        # Denominator to convert counts to rates
        denominator = dataset.acquisition_time_per_esa_step * n_sector
        # Do not carry these variable attributes from L1a to L1b for above products
        drop_variables = [
            "k_factor",
            "nso_half_spin",
            "sw_bias_gain_mode",
            "st_bias_gain_mode",
            "spin_period",
            "voltage_table",
            # TODO: undo this when I get new validation file from Joey
            # "acquisition_time_per_esa_step",
        ]
        dataset = dataset.drop_vars(drop_variables)

    elif descriptor in [
        "hi-counters-aggregated",
        "hi-counters-singles",
        "hi-omni",
        "hi-priority",
        "hi-sectored",
        "hi-ialirt",
    ]:
        # Denominator to convert counts to rates
        denominator = (
            constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spin_sectors"]
            * constants.L1B_DATA_PRODUCT_CONFIGURATIONS[descriptor]["num_spins"]
            * constants.HI_ACQUISITION_TIME
        )

    # For each variable, convert counts and uncertainty to rates
    for variable in variables_to_convert:
        dataset[variable].data = dataset[variable].astype(np.float64) / denominator
        # Carry over attrs and update as needed
        dataset[variable].attrs["UNITS"] = "counts/s"

        if calculate_unc:
            # Uncertainty calculation
            unc_variable = f"unc_{variable}"
            dataset[unc_variable].data = (
                dataset[unc_variable].astype(np.float64) / denominator
            )
            dataset[unc_variable].attrs["UNITS"] = "1/s"

    # Drop spin_period
    if "spin_period" in dataset.variables:
        dataset = dataset.drop_vars("spin_period")

    return dataset


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

    # Get the L1b CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")

    # Use the L1a data product as a starting point for L1b
    l1b_dataset = l1a_dataset.copy(deep=True)

    # Update the global attributes
    l1b_dataset.attrs = cdf_attrs.get_global_attributes(dataset_name)
    return convert_to_rates(
        l1b_dataset,
        descriptor,
    )
