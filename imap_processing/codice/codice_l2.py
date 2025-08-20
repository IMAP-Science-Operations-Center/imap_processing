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
    dataset_name = dataset_name.replace(data_level, "l2")

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

    if dataset_name in [
        "imap_codice_l2_hi-counters-singles",
        "imap_codice_l2_hi-counters-aggregated",
        "imap_codice_l2_lo-counters-singles",
        "imap_codice_l2_lo-counters-aggregated",
        "imap_codice_l2_lo-sw-priority",
        "imap_codice_l2_lo-nsw-priority",
    ]:
        # No changes needed. Just save to an L2 CDF file.
        pass

    elif dataset_name == "imap_codice_l2_hi-direct-events":
        # Convert the following data variables to physical units using
        # calibration data:
        #    - ssd_energy
        #    - tof
        #    - elevation_angle
        #    - spin_angle
        # These converted variables are *in addition* to the existing L1 variables
        # The other data variables require no changes
        # See section 11.1.2 of algorithm document
        pass

    elif dataset_name == "imap_codice_l2_hi-sectored":
        # Convert the sectored count rates using equation described in section
        # 11.1.3 of algorithm document.
        pass

    elif dataset_name == "imap_codice_l2_hi-omni":
        # Calculate the omni-directional intensity for each species using
        # equation described in section 11.1.4 of algorithm document
        # hopefully this can also apply to hi-ialirt
        pass

    elif dataset_name == "imap_codice_l2_lo-direct-events":
        # Convert the following data variables to physical units using
        # calibration data:
        #    - apd_energy
        #    - elevation_angle
        #    - tof
        #    - spin_sector
        #    - esa_step
        # These converted variables are *in addition* to the existing L1 variables
        # The other data variables require no changes
        # See section 11.1.2 of algorithm document
        pass

    elif dataset_name == "imap_codice_l2_lo-sw-angular":
        # Calculate the sunward angular intensities using equation described in
        # section 11.2.3 of algorithm document.
        pass

    elif dataset_name == "imap_codice_l2_lo-nsw-angular":
        # Calculate the non-sunward angular intensities using equation described
        # in section 11.2.3 of algorithm document.
        pass

    elif dataset_name == "imap_codice_l2_lo-sw-species":
        # Calculate the sunward solar wind species intensities using equation
        # described in section 11.2.4 of algorithm document.
        # Calculate the pickup ion sunward solar wind intensities using equation
        # described in section 11.2.4 of algorithm document.
        # Hopefully this can also apply to lo-ialirt
        pass

    elif dataset_name == "imap_codice_l2_lo-nsw-species":
        # Calculate the non-sunward solar wind species intensities using
        # equation described in section 11.2.4 of algorithm document.
        # Calculate the pickup ion non-sunward solar wind intensities using
        # equation described in section 11.2.4 of algorithm document.
        pass

    logger.info(f"\nFinal data product:\n{l2_dataset}\n")

    return l2_dataset
