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
from imap_data_access import ProcessingInputCollection

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.constants import (
    HALF_SPIN_LUT,
    HI_OMNI_VARIABLE_NAMES,
    HI_SECTORED_VARIABLE_NAMES,
    L2_GEOMETRIC_FACTOR,
    L2_HI_NUMBER_OF_SSD,
    L2_HI_SECTORED_ANGLE,
    LO_NSW_SPECIES_VARIABLE_NAMES,
    LO_SW_PICKUP_ION_SPECIES_VARIABLE_NAMES,
    LO_SW_SPECIES_VARIABLE_NAMES,
    NSW_POSITIONS,
    PUI_POSITIONS,
    SW_POSITIONS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_geometric_factor_lut(dependencies: ProcessingInputCollection) -> dict:
    """
    Get the geometric factor lookup table.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    geometric_factor_lut : dict
        A dict with a full and reduced mode array with shape (esa_steps, position).
    """
    geometric_factors = pd.read_csv(
        dependencies.get_file_paths(descriptor="l2-lo-gfactor")[0]
    )

    # sort by esa step. They should already be sorted, but just in case
    full = geometric_factors[geometric_factors["mode"] == "full"].sort_values(
        by="esa_step"
    )
    reduced = geometric_factors[geometric_factors["mode"] == "reduced"].sort_values(
        by="esa_step"
    )

    # Sort position columns to ensure the correct order
    position_names_sorted = sorted(
        [col for col in full if col.startswith("position")],
        key=lambda x: int(x.split("_")[-1]),
    )

    return {
        "full": full[position_names_sorted].to_numpy(),
        "reduced": reduced[position_names_sorted].to_numpy(),
    }


def get_efficiency_lut(dependencies: ProcessingInputCollection) -> pd.DataFrame:
    """
    Get the efficiency lookup table.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    efficiency_lut : pandas.DataFrame
        Contains the efficiency lookup table. Columns are:
        species, product, esa_step, position_1, position_2, ..., position_24.
    """
    return pd.read_csv(dependencies.get_file_paths(descriptor="l2-lo-efficiency")[0])


def get_species_efficiency(species: str, efficiency: pd.DataFrame) -> np.ndarray:
    """
    Get the efficiency values for a given species.

    Parameters
    ----------
    species : str
        The species name.
    efficiency : pandas.DataFrame
        The efficiency lookup table.

    Returns
    -------
    efficiency : np.ndarray
        A 2D array of efficiencies with shape (epoch, esa_steps).
    """
    species_efficiency = efficiency[efficiency["species"] == species].sort_values(
        by="esa_step"
    )
    # Sort position columns to ensure the correct order
    position_names_sorted = sorted(
        [col for col in species_efficiency if col.startswith("position")],
        key=lambda x: int(x.split("_")[-1]),
    )
    # Shape: (esa_steps, positions)
    return species_efficiency[position_names_sorted].to_numpy()


def compute_geometric_factors(
    dataset: xr.Dataset, geometric_factor_lookup: dict
) -> np.ndarray:
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
    geometric_factor_lookup : dict
        A dict with a full and reduced mode array with shape (esa_steps, position).

    Returns
    -------
    geometric_factors : np.ndarray
        A 3D array of geometric factors with shape (epoch, esa_steps, positions).
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
    # Perform the comparison and calculate modes
    # Modes will be true (reduced mode) anywhere half_spin >= rgfo_half_spin otherwise
    # false (full mode)
    modes = half_spin_values >= rgfo_half_spin

    # Get the geometric factors based on the modes
    gf = np.where(
        modes[:, :, np.newaxis],  # Shape (epoch, esa_step, 1)
        geometric_factor_lookup["reduced"],  # Shape (1, esa_step, 24) - reduced mode
        geometric_factor_lookup["full"],  # Shape (1, esa_step, 24) - full mode
    )  # Shape: (epoch, esa_step, positions)
    return gf


def process_lo_species_intensity(
    dataset: xr.Dataset,
    species_list: list,
    geometric_factors: np.ndarray,
    efficiency: pd.DataFrame,
    positions: list,
) -> xr.Dataset:
    """
    Process the lo-species L2 dataset to calculate species intensities.

    Parameters
    ----------
    dataset : xarray.Dataset
        The L2 dataset to process.
    species_list : list
        List of species variable names to calculate intensity.
    geometric_factors : np.ndarray
        The geometric factors array with shape (epoch, esa_steps).
    efficiency : pandas.DataFrame
        The efficiency lookup table.
    positions : list
        A list of position indices to select from the geometric factor and
        efficiency lookup tables.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with species intensities calculated.
    """
    # Select the relevant positions from the geometric factors
    geometric_factors = geometric_factors[:, :, positions]
    # take the mean geometric factor across positions
    geometric_factors = np.nanmean(geometric_factors, axis=-1)
    scaler = len(positions)
    # Calculate the species intensities using the provided geometric factors and
    # efficiency. Species_intensity = species_rate / (gm * eff * esa_step)
    for species in species_list:
        # Select the relevant positions for the species from the efficiency LUT
        # Shape: (epoch, esa_steps, positions)
        species_eff = get_species_efficiency(species, efficiency)[
            np.newaxis, :, positions
        ]
        if species_eff.size == 0:
            logger.warning("No efficiency data found for species {species}. Skipping.")
            continue
        # Take the mean efficiency across positions
        species_eff = np.nanmean(species_eff, axis=-1)
        denominator = (
            scaler * geometric_factors * species_eff * dataset["energy_table"].data
        )
        if species not in dataset:
            logger.warning(
                f"Species {species} not found in dataset. Filling with NaNS."
            )
            dataset[species] = np.full(dataset["energy_table"].data.shape, np.nan)
        else:
            dataset[species] = dataset[species] / denominator[:, :, np.newaxis]

    return dataset


def process_hi_omni(
    l2_dataset: xr.Dataset, dependencies: ProcessingInputCollection
) -> xr.Dataset:
    """
    Process the hi-omni L1B dataset to calculate omni-directional intensities.

    See section 11.1.3 of the CoDICE algorithm document for details.

    The formula for omni-directional intensities is::

        l1B species data / (
            geometric_factor * number_of_ssd * efficiency * energy_passband
        )

    Geometric factor is constant for all species which is 0.013.
    Number of SSD is constant for all species which is 12.
    Efficiency is provided in a CSV file for each species and energy bin.
    Energy passband is calculated from L1B variables energy_bin_minus + energy_bin_plus

    Parameters
    ----------
    l2_dataset : xarray.Dataset
        The L2 dataset to process.
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with omni-directional intensities calculated.
    """
    # Read the efficiencies data from the CSV file
    efficiencies_file = dependencies.get_file_paths(descriptor="l2-hi-omni-efficiency")[
        0
    ]
    efficiencies_df = pd.read_csv(efficiencies_file)
    # Omni product has 8 species and each species has different shape.
    # Eg.
    #   h - (epoch, 15)
    #   c - (epoch, 18)
    #   uh - (epoch, 5)
    #   etc.
    # Because of that, we need to loop over each species and calculate
    # omni-directional intensities separately.
    for species in HI_OMNI_VARIABLE_NAMES:
        species_data = efficiencies_df[efficiencies_df["species"] == species]
        # Read current species' effificiency
        species_efficiencies = species_data["average_efficiency"].values[np.newaxis, :]
        # Calculate energy passband from L1B data
        energy_passbands = (
            l2_dataset[f"energy_{species}_plus"] + l2_dataset[f"energy_{species}_minus"]
        ).values[np.newaxis, :]
        # Calculate omni-directional intensities
        omni_direction_intensities = l2_dataset[species] / (
            L2_GEOMETRIC_FACTOR
            * L2_HI_NUMBER_OF_SSD
            * species_efficiencies
            * energy_passbands
        )
        # Store by replacing existing species data with omni-directional intensities
        l2_dataset[species].values = omni_direction_intensities

    return l2_dataset


def process_hi_sectored(
    l2_dataset: xr.Dataset, dependencies: ProcessingInputCollection
) -> xr.Dataset:
    """
    Process the hi-omni L1B dataset to calculate omni-directional intensities.

    See section 11.1.2 of the CoDICE algorithm document for details.

    The formula for omni-directional intensities is::

        l1b species data / (geometric_factor * efficiency * energy_passband)

    Geometric factor is constant for all species and is 0.013.
    Efficiency is provided in a CSV file for each species and energy bin and
    position.
    Energy passband is calculated from energy_bin_minus + energy_bin_plus

    Parameters
    ----------
    l2_dataset : xarray.Dataset
        The L2 dataset to process.
    dependencies : ProcessingInputCollection
        The collection of processing input files.

    Returns
    -------
    xarray.Dataset
        The updated L2 dataset with omni-directional intensities calculated.
    """
    efficiencies_file = dependencies.get_file_paths(
        descriptor="l2-hi-sectored-efficiency"
    )[0]
    efficiencies_df = pd.read_csv(efficiencies_file)
    # Similar to hi-omni, each species has different shape.
    # Because of that, we need to loop over each species and calculate
    # sectored intensities separately.
    for species in HI_SECTORED_VARIABLE_NAMES:
        # Efficiencies from dataframe maps to different dimension in L1B data.
        # For example:
        #   l1b species 'h' has shape:
        #       (epoch, 8, 12, 12) -> (time, energy, spin_sector, inst_az)
        #   efficiencies 'h' has shape after reading from CSV:
        #       (8, 12) -> (energy, inst_az)
        #       NOTE: 12 here maps to last 12 in above l1b dimension.
        # Because of this, it's easier to work with the data in xarray.
        # Xarray automatically aligns dimensions and coordinates, making it easier
        # to work with multi-dimensional data. Thus, we convert the efficiencies
        # to xarray.DataArray with dimensions (energy, ssd_index)
        # TODO: update ssd_index to inst_az when Joey data is updated.
        species_data = efficiencies_df[efficiencies_df["species"] == species].values
        species_efficiencies = xr.DataArray(
            species_data[:, 2:].astype(
                float
            ),  # Skip first two columns (species, energy_bin)
            dims=(f"energy_{species}", "ssd_index"),
            coords=l2_dataset[[f"energy_{species}", "ssd_index"]],
        )

        # energy_passbands has shape:
        #   (8,) -> (energy)
        energy_passbands = xr.DataArray(
            l2_dataset[f"energy_{species}_minus"]
            + l2_dataset[f"energy_{species}_plus"],
            dims=(f"energy_{species}",),
            coords=l2_dataset[[f"energy_{species}"]],
            name="passband",
        )

        sectored_intensities = l2_dataset[species] / (
            L2_GEOMETRIC_FACTOR * species_efficiencies * energy_passbands
        )

        # Replace existing species data with omni-directional intensities
        l2_dataset[species].values = sectored_intensities

    # Calculate spin angle
    # Formula:
    #   θ_(k,n) = (θ_(k,0)+30°* n)  mod 360°
    # where
    #   n is size of L2_HI_SECTORED_ANGLE, 0 to 11,
    #   k is size of ssd_index from l1b, 0 to 11,
    # Calculate spin angle by adding a base angle from L2_HI_SECTORED_ANGLE
    # for each SSD index and then adding multiple of 30 degrees for each elevation.
    # Then mod by 360 to keep it within 0-360 range.
    elevation_angles = np.arange(len(l2_dataset["ssd_index"].values)) * 30.0
    spin_angles = (L2_HI_SECTORED_ANGLE[:, np.newaxis] + elevation_angles) % 360.0
    # TODO: add CDF attrs
    l2_dataset["spin_angles"] = (("spin_sector", "elevation_angle"), spin_angles)
    return l2_dataset


def process_codice_l2(
    file_path: Path, dependencies: ProcessingInputCollection
) -> xr.Dataset:
    """
    Will process CoDICE l1 data to create l2 data products.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the CoDICE L1 file to process.
    dependencies : ProcessingInputCollection
        Collection of processing inputs such as ancillary data files.

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
    # cdf_attrs = ImapCdfAttributes()

    # TODO uncomment and update variable attrs
    # l2_dataset = add_dataset_attributes(l2_dataset, dataset_name, cdf_attrs)

    # TODO: update list of datasets that need geometric factors (if needed)
    # Compute geometric factors needed for intensity calculations
    if dataset_name in [
        "imap_codice_l2_lo-sw-species",
        "imap_codice_l2_lo-nsw-species",
    ]:
        geometric_factor_lookup = get_geometric_factor_lut(dependencies)
        efficiency_lookup = get_efficiency_lut(dependencies)
        geometric_factors = compute_geometric_factors(
            l2_dataset, geometric_factor_lookup
        )

        if dataset_name == "imap_codice_l2_lo-sw-species":
            # Filter the efficiency lookup table for solar wind efficiencies
            efficiencies = efficiency_lookup[efficiency_lookup["product"] == "sw"]
            # Calculate the pickup ion sunward solar wind intensities using equation
            # described in section 11.2.4 of algorithm document.
            process_lo_species_intensity(
                l2_dataset,
                LO_SW_PICKUP_ION_SPECIES_VARIABLE_NAMES,
                geometric_factors,
                efficiencies,
                PUI_POSITIONS,
            )
            # Calculate the sunward solar wind species intensities using equation
            # described in section 11.2.4 of algorithm document.
            process_lo_species_intensity(
                l2_dataset,
                LO_SW_SPECIES_VARIABLE_NAMES,
                geometric_factors,
                efficiencies,
                SW_POSITIONS,
            )
        else:
            # Filter the efficiency lookup table for non solar wind efficiencies
            efficiencies = efficiency_lookup[efficiency_lookup["product"] == "nsw"]
            # Calculate the non-sunward species intensities using equation
            # described in section 11.2.4 of algorithm document.
            process_lo_species_intensity(
                l2_dataset,
                LO_NSW_SPECIES_VARIABLE_NAMES,
                geometric_factors,
                efficiencies,
                NSW_POSITIONS,
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
        process_hi_sectored(l2_dataset, dependencies)

    elif dataset_name == "imap_codice_l2_hi-omni":
        # Calculate the omni-directional intensity for each species using
        # equation described in section 11.1.4 of algorithm document
        # hopefully this can also apply to hi-ialirt
        process_hi_omni(l2_dataset, dependencies)

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
