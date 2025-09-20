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
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.constants import (
    HALF_SPIN_LUT,
    HI_ELEVATION_ANGLE,
    HI_SPIN_ANGLE,
    HI_SSD_ID_TO_INDEX,
)
from imap_processing.codice.utils import reshape_ssd_energy_df

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def codice_hi_direct_event(l1a_data: xr.Dataset) -> xr.Dataset:
    """
    Calculate Hi Direct Event physical quantities.

    Convert the following data variables to physical units using
    calibration data:
    - ssd_energy
    - tof
    - elevation_angle
    - spin_angle

    The other data variables require no changes.
    See section 11.1.2 of algorithm document.

    Parameters
    ----------
    l1a_data : xarray.Dataset
        The input dataset containing the raw data variables.

    Returns
    -------
    data : xarray.Dataset
        The dataset with the physical quantities calculated.
    """
    # ---------------------------------------------------
    # Carry these data in L2 without any modification
    #   Gain
    #   Multi-Flag
    #   Spin Number
    # ---------------------------------------------------
    l2_vars = [var for var in ["gain", "multi_flag", "spin_number"] if var in l1a_data]
    l2_dataset = l1a_data[l2_vars].copy(deep=True)

    # ------------------------------------------------------
    # SSD Energy
    # ------------------------------------------------------
    # Read in SSD energy value
    ssd_energy_file = Path(
        f"{imap_module_directory}/"
        "codice/data/imap_codice_l2_hi-ssd-energy-lut_2050523_v001.csv"
    )
    ssd_energy_df = pd.read_csv(ssd_energy_file)

    # LUT table map SSD ID and Gain to column names.
    # Formula to lookup correct column name in LUT is
    #   SSD_ID * 3 + Gain - 1
    # Example of column name: SSD 1 - MG. But we
    # reshape the dataframe into a 3D array (rows, ssd, gain)
    # This makes lookup by SSD ID and gain more efficient
    # - First dimension: bin_num (row index in the CSV)
    # - Second dimension: SSD ID (0-15)
    # - Third dimension: Gain (0=LG, 1=MG, 2=HG)
    lut_ssd_energy_3d = reshape_ssd_energy_df(ssd_energy_df)

    # ssd_energy shape (n, priority, event_num)
    l1a_ssd_energy = l1a_data["ssd_energy"].data
    l1a_ssd_id = l1a_data["ssd_id"].data
    # 'gain' variable has 0 to 3 where
    #  0	No energy
    #  1	Low Gain or LG
    #  2	Mid Gain or MG
    #  3	High Gain or HG

    l1a_gain = l1a_data["gain"].data

    # Flatten the arrays for vectorized operations
    flat_ssd_id = l1a_ssd_id.flatten()
    flat_gain = l1a_gain.flatten()

    # Create mask for valid values
    #   SSD ID must be <= 15
    #   Gain must be > 0 and <= 3 (1=LG, 2=MG, 3=HG)
    #   SSD Energy must be > 0 and <= 2047
    # These variables map to those number in LUT
    flat_ssd_energy = l1a_ssd_energy.flatten()
    valid_mask = (
        (flat_ssd_id <= 15)
        & (flat_gain > 0)
        & (flat_gain <= 3)
        & (flat_ssd_energy <= 2047)
        & (flat_ssd_energy > 0)
    )

    # Create arrays to store results
    l2_ssd_energy = np.full(flat_ssd_id.shape, np.nan, dtype=float)

    # First, Store invalid indices for later
    invalid_indices = np.where(~valid_mask)[0]

    # Set invalid values to zero (or valid defaults) for safe lookup.
    # This is workaround to not raise errors during lookup. We
    # fill invalid values with nan in later step.
    flat_ssd_energy[~valid_mask] = 0
    flat_ssd_id[~valid_mask] = 0
    flat_gain[~valid_mask] = 0

    # Look up all values at once. This is looking up
    # (row, column, cell) using (ssd_energy, ssd_id, gain)
    # using those variable's value from l1a.
    l2_ssd_energy = lut_ssd_energy_3d[flat_ssd_energy, flat_ssd_id, flat_gain]

    # Lastly, fill invalid indices back with NaN
    l2_ssd_energy[invalid_indices] = np.nan

    l2_dataset["ssd_energy"] = (
        l1a_data["ssd_energy"].dims,
        l2_ssd_energy.reshape(l1a_ssd_energy.shape),
    )

    # ------------------------------------
    # TOF in ns
    # ------------------------------------
    # LUT has these column, tof_raw,TOF (ns),E/n (MeV/n)
    tof_file = Path(
        f"{imap_module_directory}/"
        "codice/data/imap_codice_l2_hi-tof-lut_2050523_v001.csv"
    )
    tof_df = pd.read_csv(tof_file)
    # Read LUT column into its variables
    lut_raw = tof_df["tof_raw"].values
    lut_tof_ns = tof_df["TOF (ns)"].values

    # L1A TOF data shape (n, 6, event_num)
    l1a_tof_data = l1a_data["tof"].data

    # Flatten l1a_tof_data for easier look up
    l1a_tof_data_flat = l1a_tof_data.flatten()

    # Only process valid TOF values (<= 1023):
    #   * Create a mask for valid TOF values
    #   * Prepare integer index array with -1 default
    #   * Look up where each valid TOF using searchsorted
    valid_mask = l1a_tof_data_flat <= 1023
    l1a_tof_data_idx = np.full_like(l1a_tof_data_flat, -1, dtype=int)
    l1a_tof_data_idx[valid_mask] = np.searchsorted(
        lut_raw, l1a_tof_data_flat[valid_mask]
    )
    # If TOF data has value greater than 1023, fill nan
    # by default
    tof_ns = np.full_like(l1a_tof_data_flat, np.nan, dtype=float)
    # Map indices to TOF (ns) data for valid entries
    tof_ns[valid_mask] = lut_tof_ns[l1a_tof_data_idx[valid_mask]]
    # Reshape back to original shape
    tof_ns = tof_ns.reshape(l1a_tof_data.shape)

    # Add to dataset as new variable
    l2_dataset["tof"] = (l1a_data["tof"].dims, tof_ns)

    # ------------------------
    # Elevation and Spin Angle
    # ------------------------
    # SSD ID is used for finding both elevation and spin angles
    ssd_id_flat = l1a_ssd_id.flatten()

    # Map SSD ID to index. If bigger than 2047, it's solar anomalies. Use nan as
    # indicator of solar anomalies
    ssd_idx = np.array([HI_SSD_ID_TO_INDEX.get(x, np.nan) for x in ssd_id_flat])
    # Prepare mask for valid indices that are not nan to look up
    # elevation angle
    elevation_valid_mask = ~np.isnan(ssd_idx)
    elevation_angle = np.full_like(ssd_idx, np.nan, dtype=float)
    elevation_angle[elevation_valid_mask] = HI_ELEVATION_ANGLE[
        ssd_idx[elevation_valid_mask].astype(int)
    ]
    elevation_angle = elevation_angle.reshape(l1a_ssd_id.shape)
    l2_dataset["elevation_angle"] = (l1a_data["ssd_id"].dims, elevation_angle)

    # Prepare mask for valid indices that are not nan to look up
    # spin angle
    spin_valid_mask = ~np.isnan(ssd_idx)
    spin_angle = np.full_like(ssd_idx, np.nan, dtype=float)
    spin_angle[spin_valid_mask] = HI_SPIN_ANGLE[ssd_idx[spin_valid_mask].astype(int)]
    spin_angle = spin_angle.reshape(l1a_ssd_id.shape)
    l2_dataset["spin_angle"] = (l1a_data["ssd_id"].dims, spin_angle)

    return l2_dataset


def process_codice_l2(file_path: Path) -> xr.Dataset:
    """
    Process L1A Direct Events to L2.

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
        l2_dataset = codice_hi_direct_event(l2_dataset)

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

    logger.info(f"\nProcessing completed: {dataset_name}")

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
