"""IMAP-HIT L2 data processing."""

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.hit.hit_utils import (
    add_energy_variables,
    get_attribute_manager,
    initialize_particle_data_arrays,
    sum_particle_data,
)
from imap_processing.hit.l2.constants import (
    L2_STANDARD_ANCILLARY_PATH_PREFIX,
    L2_SUMMED_ANCILLARY_PATH_PREFIX,
    STANDARD_PARTICLE_ENERGY_RANGE_MAPPING,
)

logger = logging.getLogger(__name__)

# TODO:
#  - review logging levels to use (debug vs. info)
#  - determine where to pull ancillary data. Storing it locally for now


def hit_l2(dependency: xr.Dataset, data_version: str) -> list[xr.Dataset]:
    """
    Will process HIT data to L2.

    Processes dependencies needed to create L2 data products.

    Parameters
    ----------
    dependency : xr.Dataset
        L1B xarray science dataset that is either summed rates
        standard rates or sector rates.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of one L2 dataset.
    """
    logger.info("Creating HIT L2 science datasets")
    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager(data_version, "l2")

    # TODO: Write functions to process sectored rates dataset
    #       with logical source: "imap_hit_l2_macropixel-intensity"

    l2_datasets: dict = {}

    # Process science data to L2 datasets
    if "imap_hit_l1b_summed-rates" in dependency.attrs["Logical_source"]:
        l2_datasets["imap_hit_l2_summed-intensity"] = process_summed_intensity_data(
            dependency
        )
        logger.info("HIT L2 summed intensity dataset created")

    if "imap_hit_l1b_standard-rates" in dependency.attrs["Logical_source"]:
        l2_datasets["imap_hit_l2_standard-intensity"] = process_standard_intensity_data(
            dependency
        )
        logger.info("HIT L2 standard intensity dataset created")

    # Update attributes and dimensions
    for logical_source, dataset in l2_datasets.items():
        dataset.attrs = attr_mgr.get_global_attributes(logical_source)

        # TODO: Add CDF attributes to yaml once they're defined for L2 science data
        #  consider moving attribute handling to hit_utils.py
        # Assign attributes and dimensions to each data array in the Dataset
        for field in dataset.data_vars.keys():
            try:
                # Create a dict of dimensions using the DEPEND_I keys in the
                # attributes
                dims = {
                    key: value
                    for key, value in attr_mgr.get_variable_attributes(field).items()
                    if "DEPEND" in key
                }
                dataset[field].attrs = attr_mgr.get_variable_attributes(field)
                dataset[field].assign_coords(dims)
            except KeyError:
                # TODO: consider raising an error after L2 attributes are defined.
                #  Until then, continue with processing and log warning
                logger.warning(f"Field {field} not found in attribute manager.")

        # Skip schema check for epoch to prevent attr_mgr from adding the
        # DEPEND_0 attribute which isn't required for epoch
        dataset.epoch.attrs = attr_mgr.get_variable_attributes(
            "epoch", check_schema=False
        )

        logger.info(f"HIT L2 dataset created for {logical_source}")

    return list(l2_datasets.values())


class IntensityFactors(NamedTuple):
    """A namedtuple to store factors for the intensity equation."""

    delta_e_factor: np.ndarray
    geometry_factor: np.ndarray
    efficiency: np.ndarray
    b: np.ndarray


def get_intensity_factors(
    energy_min: np.ndarray, species_ancillary_data: pd.DataFrame
) -> IntensityFactors:
    """
    Get the intensity factors for all energy bins of the given species ancillary data.

    This function gets the factors needed for the equation to convert rates to
    intensities for all energy bins for the given species.

    Parameters
    ----------
    energy_min : np.ndarray
        All energy min values for the species.
    species_ancillary_data : pd.DataFrame
        The subset of ancillary data for the given species.

    Returns
    -------
    IntensityFactors
        The factors needed to convert rates to intensities for all energy bins
        for the given species.
    """
    # Get factors needed to convert rates to intensities for
    # all energy bins for the given species ancillary data
    intensity_factors = species_ancillary_data.set_index(
        species_ancillary_data["lower energy (mev)"].astype(np.float32)
    ).loc[energy_min]

    return IntensityFactors(
        delta_e_factor=intensity_factors["delta e (mev)"].values,
        geometry_factor=intensity_factors["geometry factor (cm2 sr)"].values,
        efficiency=intensity_factors["efficiency"].values,
        b=intensity_factors["b"].values,
    )


def calculate_intensities(
    rate: xr.DataArray,
    delta_e_factor: np.ndarray,
    geometry_factor: np.ndarray,
    efficiency: np.ndarray,
    b: np.ndarray,
) -> xr.DataArray:
    """
    Calculate the intensities for given arrays of rates and ancillary factors.

    Uses vectorization to calculate the intensities for an array of rates
    at an epoch.

        This function uses equation 9 and 11 from the HIT algorithm document:
        (Summed L1B Rates) / (60 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    rate : xr.DataArray
        The L1B rates to be converted to intensities for an epoch.
    delta_e_factor : np.ndarray
        The energy bin width factors for an epoch.
    geometry_factor : np.ndarray
        The geometry factors for an epoch.
    efficiency : np.ndarray
        The efficiency factors for an epoch.
    b : np.ndarray
        The b factors for an epoch.

    Returns
    -------
    xr.DataArray
        The calculated intensities for an epoch.
    """
    # Calculate the intensities for this energy bin using vectorization
    return (rate / (60 * delta_e_factor * geometry_factor * efficiency)) - b


def calculate_intensities_for_a_species(
    species_variable: str, l2_dataset: xr.Dataset, ancillary_data_frames: dict
) -> None:
    """
    Calculate the intensity for a given species in the dataset.

    Parameters
    ----------
    species_variable : str
        The species variable to calculate the intensity for which is either the species
        or a statistical uncertainty. (i.e. "h", "h_delta_minus", or "h_delta_plus").
    l2_dataset : xr.Dataset
        The L2 dataset containing the summed L1B rates to calculate the intensity.
    ancillary_data_frames : dict
        Dictionary containing ancillary data for each dynamic threshold state where
        the key is the dynamic threshold state and the value is a pandas DataFrame
        containing the ancillary data.
    """
    species = (
        species_variable.split("_")[0]
        if "_delta_" in species_variable
        else species_variable
    )
    energy_min = (
        l2_dataset[f"{species}_energy_mean"].values
        - l2_dataset[f"{species}_energy_delta_minus"].values
    )
    # TODO: Add check for energy max after ancillary file is updated
    #  to fix errors

    # Calculate the intensity for each epoch and energy bin since the
    # dynamic threshold state can vary by epoch and that determines the
    # ancillary data to use.
    for epoch in range(l2_dataset[species_variable].shape[0]):
        # Get ancillary data using the dynamic threshold state for this epoch
        species_ancillary_data = get_species_ancillary_data(
            int(l2_dataset["dynamic_threshold_state"][epoch].values),
            ancillary_data_frames,
            species,
        )

        # Calculate the intensity for this energy bin using vectorization
        # and replace rates with intensities in the dataset
        factors: IntensityFactors = get_intensity_factors(
            energy_min, species_ancillary_data
        )
        rates: xr.DataArray = l2_dataset[species_variable][epoch]

        l2_dataset[species_variable][epoch] = calculate_intensities(
            rates,
            factors.delta_e_factor,
            factors.geometry_factor,
            factors.efficiency,
            factors.b,
        )


def calculate_intensities_for_all_species(
    l2_dataset: xr.Dataset, ancillary_data_frames: dict
) -> None:
    """
    Calculate the intensity for each species in the dataset.

    Parameters
    ----------
    l2_dataset : xr.Dataset
        The L2 dataset.
    ancillary_data_frames : dict
        Dictionary containing ancillary data for each dynamic threshold state
        where the key is the dynamic threshold state and the value is a pandas
        DataFrame containing the ancillary data.
    """
    # TODO: update to also calculate intensity for sectorates?
    # List of valid species data variables to calculate intensity for
    valid_data_variables = [
        "h",
        "he3",
        "he4",
        "he",
        "c",
        "n",
        "o",
        "ne",
        "na",
        "mg",
        "al",
        "si",
        "s",
        "ar",
        "ca",
        "fe",
        "ni",
    ]

    # Add statistical uncertainty variables to the list of valid variables
    valid_data_variables += [f"{var}_delta_minus" for var in valid_data_variables] + [
        f"{var}_delta_plus" for var in valid_data_variables
    ]

    # Calculate the intensity for each valid data variable
    for species_variable in valid_data_variables:
        if species_variable in l2_dataset.data_vars:
            calculate_intensities_for_a_species(
                species_variable, l2_dataset, ancillary_data_frames
            )
        else:
            logger.warning(
                f"Variable {species_variable} not found in dataset. "
                f"Skipping intensity calculation."
            )


def add_systematic_uncertainties(
    dataset: xr.Dataset, particle: str, energy_bins: int
) -> None:
    """
    Add systematic uncertainties to the dataset.

    Add systematic uncertainties to the dataset. Just zeros for now.
    To change if/when HIT determines there are systematic uncertainties.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to add the systematic uncertainties to.
    particle : str
        The particle name.
    energy_bins : int
        Number of energy bins for the particle.
    """
    dataset[f"{particle}_sys_delta_minus"] = xr.DataArray(
        data=np.zeros(energy_bins, dtype=np.float32),
        dims=[f"{particle}_energy_mean"],
        name=f"{particle}_sys_delta_minus",
    )
    dataset[f"{particle}_sys_delta_plus"] = xr.DataArray(
        data=np.zeros(energy_bins, dtype=np.float32),
        dims=[f"{particle}_energy_mean"],
        name=f"{particle}_sys_delta_plus",
    )


def add_standard_particle_rates_to_dataset(
    l2_standard_intensity_dataset: xr.Dataset,
    l1b_standard_rates_dataset: xr.Dataset,
    particle: str,
    energy_ranges: list,
) -> None:
    """
    Add summed standard particle rates to the dataset.

    This function performs the following steps:
      1) sum the standard rates, including statistical uncertainties,
         from the l2fgrates, l3fgrates, and penfgrates data variables in the L1B
         standard rates data.
      2) add the summed rates to the L2 standard intensity dataset by particle type
         and energy range.

    Parameters
    ----------
    l2_standard_intensity_dataset : xr.Dataset
        The L2 standard intensity dataset to add the rates to.
    l1b_standard_rates_dataset : xr.Dataset
        The L1B standard rates dataset containing rates to sum.
    particle : str
        The particle name.
    energy_ranges : list
        A list of energy range dictionaries for the particle.
        For example:
        {'energy_min': 1.8, 'energy_max': 2.2, "R2": [1], "R3": [], "R4": []}.
    """
    # Initialize arrays to store summed rates and statistical uncertainties
    l2_standard_intensity_dataset = initialize_particle_data_arrays(
        l2_standard_intensity_dataset,
        particle,
        len(energy_ranges),
        l1b_standard_rates_dataset.sizes["epoch"],
    )

    # initialize arrays to store energy min and max values
    energy_min = np.zeros(len(energy_ranges), dtype=np.float32)
    energy_max = np.zeros(len(energy_ranges), dtype=np.float32)

    # Sum particle rates and statistical uncertainties for each energy range
    # and add them to the dataset
    for i, energy_range_dict in enumerate(energy_ranges):
        summed_rates, summed_rates_delta_minus, summed_rates_delta_plus = (
            sum_particle_data(l1b_standard_rates_dataset, energy_range_dict)
        )

        l2_standard_intensity_dataset[f"{particle}"][:, i] = summed_rates.astype(
            np.float32
        )
        l2_standard_intensity_dataset[f"{particle}_delta_minus"][:, i] = (
            summed_rates_delta_minus.astype(np.float32)
        )
        l2_standard_intensity_dataset[f"{particle}_delta_plus"][:, i] = (
            summed_rates_delta_plus.astype(np.float32)
        )

        # Fill energy min and max values for each energy range
        energy_min[i] = energy_range_dict["energy_min"]
        energy_max[i] = energy_range_dict["energy_max"]

    l2_standard_intensity_dataset = add_energy_variables(
        l2_standard_intensity_dataset, particle, energy_min, energy_max
    )


def get_species_ancillary_data(
    dynamic_threshold_state: int, ancillary_data_frames: dict, species: str
) -> pd.DataFrame:
    """
    Get the ancillary data for a given species and dynamic threshold state.

    Parameters
    ----------
    dynamic_threshold_state : int
        The dynamic threshold state for the ancillary data (0-3).
    ancillary_data_frames : dict
        Dictionary containing ancillary data for each dynamic threshold state
        where the key is the dynamic threshold state and the value is a pandas
        DataFrame containing the ancillary data.
    species : str
        The species to get the ancillary data for.

    Returns
    -------
    pd.DataFrame
        The ancillary data for the species and dynamic threshold state.
    """
    ancillary_data = ancillary_data_frames[dynamic_threshold_state]

    # Get the ancillary data for the species
    species_ancillary_data = ancillary_data[ancillary_data["species"] == species]
    return species_ancillary_data


def load_ancillary_data(dynamic_threshold_states: set, path_prefix: Path) -> dict:
    """
    Load ancillary data based on the dynamic threshold state.

    The dynamic threshold state (0-3) determines which ancillary file to use.
    This function returns a dictionary with ancillary data for each state in
    the dataset.

    Parameters
    ----------
    dynamic_threshold_states : set
        A set of dynamic threshold states in the L2 dataset.
    path_prefix : Path
        The path prefix for ancillary data files.

    Returns
    -------
    dict
        A dictionary with ancillary data for each dynamic threshold state.
    """
    # Load ancillary data
    ancillary_data_frames = {
        int(state): pd.read_csv(f"{path_prefix}{state}-factors_20250219_v002.csv")
        for state in dynamic_threshold_states
    }

    # Convert column names and species values to lowercase
    for df in ancillary_data_frames.values():
        df.columns = df.columns.str.lower().str.strip()
        df["species"] = df["species"].str.lower()

    return ancillary_data_frames


def process_summed_intensity_data(l1b_summed_rates_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process L2 HIT summed intensity data from L1B summed rates.

    This function converts the L1B summed rates to L2 summed intensities
    using ancillary tables containing factors needed to calculate the
    intensity (energy bin width, geometry factor, efficiency, and b).

    Equation 12 from the HIT algorithm document:
    Summed Intensity = (L1B Summed Rate) /
                       (60 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    l1b_summed_rates_dataset : xarray.Dataset
        HIT L1B summed rates dataset.

    Returns
    -------
    xr.Dataset
        The processed L2 summed intensity dataset.
    """
    # Create a new dataset to store the L2 summed intensity data
    l2_summed_intensity_dataset = l1b_summed_rates_dataset.copy(deep=True)

    # Load ancillary data for each dynamic threshold state into a dictionary
    ancillary_data_frames = load_ancillary_data(
        set(l2_summed_intensity_dataset["dynamic_threshold_state"].values),
        L2_SUMMED_ANCILLARY_PATH_PREFIX,
    )

    # Add systematic uncertainties and energy variables to the dataset
    for var in l2_summed_intensity_dataset.data_vars:
        if "_" not in var:
            particle = str(var)
            # Add systematic uncertainties to the dataset. These will not have the
            # intensity calculation applied to them and values will be zeros
            add_systematic_uncertainties(
                l2_summed_intensity_dataset,
                particle,
                l2_summed_intensity_dataset[var].shape[1],
            )

            # TODO: remove this code after L1B is updated to have energy mean and deltas
            #       instead of energy index, min, and max
            # Add energy variables to the dataset (energy mean and deltas)
            l2_summed_intensity_dataset = add_energy_variables(
                l2_summed_intensity_dataset,
                particle,
                l2_summed_intensity_dataset[f"{particle}_energy_min"].values,
                l2_summed_intensity_dataset[f"{particle}_energy_max"].values,
            )

            # Replace energy index with energy mean as a coordinate
            l2_summed_intensity_dataset = l2_summed_intensity_dataset.assign_coords(
                {
                    f"{particle}_energy_mean": (
                        f"{particle}_energy_index",
                        l2_summed_intensity_dataset[f"{particle}_energy_mean"].values,
                    )
                }
            ).swap_dims({f"{particle}_energy_index": f"{particle}_energy_mean"})

            # Drop energy min, max, and index variables
            l2_summed_intensity_dataset = l2_summed_intensity_dataset.drop_vars(
                [
                    f"{particle}_energy_min",
                    f"{particle}_energy_max",
                    f"{particle}_energy_index",
                ]
            )

    calculate_intensities_for_all_species(
        l2_summed_intensity_dataset, ancillary_data_frames
    )

    return l2_summed_intensity_dataset


def process_standard_intensity_data(
    l1b_standard_rates_dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Will process L2 standard intensity data from L1B standard rates data.

    This function converts L1B standard rates to L2 standard intensities for each
    particle type and energy range using ancillary tables containing factors
    needed to calculate the intensity (energy bin width, geometry factor, efficiency
    and b).

    First, rates from the l2fgrates, l3fgrates, and penfgrates data variables
    in the L1B standard rates data are summed. These variables represent rates
    for different detector penetration ranges (Range 2, Range 3, and Range 4
    respectively). Only the energy ranges specified in the
    STANDARD_PARTICLE_ENERGY_RANGE_MAPPING dictionary are included in this
    product.

    Intensity is then calculated from the summed rates using the following equation:

        Equation 10 from the HIT algorithm document:
        Standard Intensity = (Summed L1B Standard Rates) /
                             (60 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    l1b_standard_rates_dataset : xr.Dataset
        The L1B standard rates dataset.

    Returns
    -------
    xr.Dataset
        The L2 standard intensity dataset.
    """
    # Create a new dataset to store the L2 standard intensity data
    l2_standard_intensity_dataset = xr.Dataset()

    # Assign the epoch coordinate from the l1B dataset
    l2_standard_intensity_dataset = l2_standard_intensity_dataset.assign_coords(
        {"epoch": l1b_standard_rates_dataset.coords["epoch"]}
    )

    # Add dynamic threshold state to the dataset
    l2_standard_intensity_dataset["dynamic_threshold_state"] = (
        l1b_standard_rates_dataset["dynamic_threshold_state"]
    )
    l2_standard_intensity_dataset[
        "dynamic_threshold_state"
    ].attrs = l1b_standard_rates_dataset["dynamic_threshold_state"].attrs

    # Load ancillary data for each dynamic threshold state into a dictionary
    ancillary_data_frames = load_ancillary_data(
        set(l2_standard_intensity_dataset["dynamic_threshold_state"].values),
        L2_STANDARD_ANCILLARY_PATH_PREFIX,
    )

    # Process each particle type and energy range and add rates and uncertainties
    # to the dataset
    for particle, energy_ranges in STANDARD_PARTICLE_ENERGY_RANGE_MAPPING.items():
        # Add systematic uncertainties to the dataset. These will not have the intensity
        # calculation applied to them and values will be zeros
        add_systematic_uncertainties(
            l2_standard_intensity_dataset, particle, len(energy_ranges)
        )
        # Add standard particle rates and statistical uncertainties to the dataset
        add_standard_particle_rates_to_dataset(
            l2_standard_intensity_dataset,
            l1b_standard_rates_dataset,
            particle,
            energy_ranges,
        )
    calculate_intensities_for_all_species(
        l2_standard_intensity_dataset, ancillary_data_frames
    )

    return l2_standard_intensity_dataset
