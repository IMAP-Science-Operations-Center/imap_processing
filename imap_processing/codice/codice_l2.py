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

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.constants import HALF_SPIN_LUT

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
    l2_dataset = add_dataset_attributes(l2_dataset, dataset_name, cdf_attrs)

    # TODO: update list of datasets that need geometric factors (if needed)
    # Compute geometric factors needed for intensity calculations
    if dataset_name in [
        "imap_codice_l2_lo-sw-species",
        "imap_codice_l2_lo-nsw-species",
    ]:
        geometric_factors = compute_geometric_factors(l2_dataset)

    if dataset_name in [
        "imap_codice_l2_hi-counters-singles",
        "imap_codice_l2_hi-counters-aggregated",
        "imap_codice_l2_lo-counters-singles",
        "imap_codice_l2_lo-counters-aggregated",
        "imap_codice_l2_lo-sw-priority",
        "imap_codice_l2_lo-nsw-priority",
    ]:
        # No changes needed. Just save to an L2 CDF file.
        # TODO: May not even need L2 files for these products
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
        # TODO: WIP - needs to be completed
        l2_dataset = process_lo_sw_species(l2_dataset, geometric_factors)
        pass

    elif dataset_name == "imap_codice_l2_lo-nsw-species":
        # Calculate the non-sunward solar wind species intensities using
        # equation described in section 11.2.4 of algorithm document.
        # Calculate the pickup ion non-sunward solar wind intensities using
        # equation described in section 11.2.4 of algorithm document.
        pass

    logger.info(f"\nFinal data product:\n{l2_dataset}\n")

    return l2_dataset


def add_dataset_attributes(
    dataset: xr.Dataset, dataset_name: str, cdf_attrs: ImapCdfAttributes
) -> xr.Dataset:
    """
    Add the global and variable attributes to the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to update.
    dataset_name : str
        The name of the dataset.
    cdf_attrs : ImapCdfAttributes
        The attribute manager for CDF attributes.

    Returns
    -------
    xarray.Dataset
        The updated dataset.
    """
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l2")

    # Update the global attributes
    dataset.attrs = cdf_attrs.get_global_attributes(dataset_name)

    # Set the variable attributes
    for variable_name in dataset.data_vars.keys():
        try:
            dataset[variable_name].attrs = cdf_attrs.get_variable_attributes(
                variable_name, check_schema=False
            )
        except KeyError:
            # Some variables may have a product descriptor prefix in the
            # cdf attributes key if they are common to multiple products.
            descriptor = dataset_name.split("imap_codice_l2_")[-1]
            cdf_attrs_key = f"{descriptor}-{variable_name}"
            try:
                dataset[variable_name].attrs = cdf_attrs.get_variable_attributes(
                    f"{cdf_attrs_key}", check_schema=False
                )
            except KeyError:
                logger.error(
                    f"Field '{variable_name}' and '{cdf_attrs_key}' not found in "
                    f"attribute manager."
                )
    return dataset


def compute_geometric_factors(dataset: xr.Dataset) -> np.ndarray:
    """
    Calculate geometric factors needed for intensity calculations.

    Geometric factors are determined by comparing the half-spin values per
    esa_step in the HALF_SPIN_LUT to the rgfo_half_spin values in the provided
    L2 dataset.

    If the half-spin value is less than the corresponding rgfo_half_spin value,
    the geometric factor is set to 0.75 (full mode); otherwise, it is set to 0.5
    (reduced mode).

    NOTE: Half spin values are associated with ESA steps which corresponds to the
    index of the energy_per_charge dimension that is between 0 and 127.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L2 dataset containing rgfo_half_spin data variable.

    Returns
    -------
    geometric_factors : np.ndarray
        A 2D array of geometric factors with shape (epoch, esa_steps).
    """
    # Convert the HALF_SPIN_LUT to a reverse mapping of esa_step to half_spin
    esa_step_to_half_spin_map = {
        val: key for key, vals in HALF_SPIN_LUT.items() for val in vals
    }

    # Create a list of half_spin values corresponding to ESA steps (0 to 127)
    half_spin_values = np.array(
        [esa_step_to_half_spin_map[step] for step in range(128)]
    )

    # Expand dimensions to compare each rgfo_half_spin value against
    # all half_spin_values
    rgfo_half_spin = dataset.rgfo_half_spin.data[:, np.newaxis]  # Shape: (epoch, 1)

    # Perform the comparison and calculate geometric factors
    geometric_factors = np.where(half_spin_values < rgfo_half_spin, 0.75, 0.5)

    return geometric_factors


def process_lo_sw_species(
    dataset: xr.Dataset, geometric_factors: np.ndarray
) -> xr.Dataset:
    """
    Process the lo-sw-species L2 dataset to calculate species intensities.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L2 dataset to process.
    geometric_factors : np.ndarray
        The geometric factors array with shape (epoch, esa_steps).

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with species intensities calculated.
    """
    # TODO: WIP - implement intensity calculations
    # valid_solar_wind_vars = [
    #     "hplus",
    #     "heplusplus",
    #     "cplus4",
    #     "cplus5",
    #     "cplus6",
    #     "oplus5",
    #     "oplus6",
    #     "oplus7",
    #     "oplus8",
    #     "ne",
    #     "mg",
    #     "si",
    #     "fe_loq",
    #     "fe_hiq",
    # ]
    # valid_pick_up_ion_vars = ["heplus", "cnoplus"]

    return dataset
