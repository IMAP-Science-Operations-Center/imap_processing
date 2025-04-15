"""IMAP-HIT L2 data processing."""

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.hit.hit_utils import (
    add_summed_particle_data_to_dataset,
    get_attribute_manager,
)
from imap_processing.hit.l2.constants import (
    FILLVAL_FLOAT32,
    L2_SECTORED_ANCILLARY_PATH_PREFIX,
    L2_STANDARD_ANCILLARY_PATH_PREFIX,
    L2_SUMMED_ANCILLARY_PATH_PREFIX,
    SECONDS_PER_10_MIN,
    SECONDS_PER_MIN,
    STANDARD_PARTICLE_ENERGY_RANGE_MAPPING,
    VALID_SECTORED_SPECIES,
    VALID_SPECIES,
)

logger = logging.getLogger(__name__)

# TODO:
#  - review logging levels to use (debug vs. info)
#  - determine where to pull ancillary data. Storing it locally for now
#  - add function to calculate combined uncertainty and add this to L2 datasets


def hit_l2(dependency: xr.Dataset) -> list[xr.Dataset]:
    """
    Will process HIT data to L2.

    Processes dependencies needed to create L2 data products.

    Parameters
    ----------
    dependency : xr.Dataset
        L1B xarray science dataset that is either summed rates
        standard rates or sector rates.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of one L2 dataset.
    """
    logger.info("Creating HIT L2 science datasets")

    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager("l2")

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

    if "imap_hit_l1b_sectored-rates" in dependency.attrs["Logical_source"]:
        l2_datasets["imap_hit_l2_macropixel-intensity"] = (
            process_sectored_intensity_data(dependency)
        )
        logger.info("HIT L2 macropixel intensity dataset created")

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

    delta_e: np.ndarray
    geometry_factor: np.ndarray
    efficiency: np.ndarray
    b: np.ndarray
    seconds: int


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
        delta_e=intensity_factors["delta e (mev)"].values,
        geometry_factor=intensity_factors["geometry factor (cm2 sr)"].values,
        efficiency=intensity_factors["efficiency"].values,
        b=intensity_factors["b"].values,
        seconds=SECONDS_PER_MIN,
    )


def calculate_intensities(
    rates: xr.DataArray,
    factors: IntensityFactors,
) -> xr.DataArray:
    """
    Calculate the intensities for given arrays of rates and equation factors.

    Uses vectorization to calculate the intensities for an array of rates
    for all epochs.

        This function uses equation 9 and 12 from the HIT algorithm document:
        ((Summed L1B Rates) / (Seconds * Delta E * Geometry Factor * Efficiency)) - b

    Parameters
    ----------
    rates : xr.DataArray
        The L1B rates to be converted to intensities.
    factors : IntensityFactors
        This is a named tuple containing the following fields:
        - delta_e: np.ndarray of energy bin widths
        - geometry_factor: np.ndarray of geometry factors
        - efficiency: np.ndarray of efficiency factors
        - b: np.ndarray of b values
        - seconds: integer of seconds to convert counts per integration time to counts
                per second. This is either:
                60 for standard and summed intensities
                600 for sectored intensities since integration time is over 10 minutes.

    Returns
    -------
    xr.DataArray
        The calculated intensities for all epochs.
    """
    # Unpack the factors
    delta_e = factors.delta_e
    geometry_factor = factors.geometry_factor
    efficiency = factors.efficiency
    b = factors.b
    seconds = factors.seconds

    # Calculate the intensities, skipping fill values, for all epochs
    return xr.DataArray(
        np.where(
            rates != FILLVAL_FLOAT32,
            (rates / (seconds * delta_e * geometry_factor * efficiency)) - b,
            FILLVAL_FLOAT32,
        ),
        dims=rates.dims,
    )


def calculate_intensities_for_a_species(
    species_variable: str, l2_dataset: xr.Dataset, ancillary_data_frames: dict
) -> xr.Dataset:
    """
    Calculate the intensity for a given species in the dataset.

    This function calculates the intensity for a given species in the dataset
    using ancillary data determined by the dynamic threshold state.

    The intensity is calculated using the equation:
        (L1B Rates) / (Seconds * Delta E * Geometry Factor * Efficiency) - b

        where the factors are retrieved from the ancillary data for the given species
        and dynamic threshold state.

    Parameters
    ----------
    species_variable : str
        The species variable to calculate the intensity for which is either the species
        or a statistical uncertainty. (i.e. "h", "h_delta_minus", or "h_delta_plus").
    l2_dataset : xr.Dataset
        The L2 dataset containing the L1B rates to calculate the intensity.
    ancillary_data_frames : dict
        Dictionary containing ancillary data for each dynamic threshold state where
        the key is the dynamic threshold state and the value is a pandas DataFrame
        containing the ancillary data for all species.

    Returns
    -------
    updated_ds : xr.Dataset
        The updated dataset with intensities calculated for the given species.
    """
    updated_ds = l2_dataset.copy()

    # Get the species name
    species = (
        species_variable.split("_")[0]
        if "_delta_" in species_variable
        else species_variable
    )
    # Get the energy bins for the species
    species_energy_bins = (
        updated_ds[f"{species}_energy_mean"].values
        - updated_ds[f"{species}_energy_delta_minus"].values
    )
    # TODO: Add check for energy max after ancillary file is updated
    #  fixing errors

    # Get the dynamic threshold state for all epochs (one per epoch)
    dynamic_threshold_states = updated_ds["dynamic_threshold_state"].values.astype(int)

    # Subset ancillary data by the species and map to dynamic threshold states (0-3)
    species_ancillary_data_by_state = {
        state: get_species_ancillary_data(state, ancillary_data_frames, species)
        for state in np.unique(dynamic_threshold_states)
    }

    # Retrieve intensity calculation factors from ancillary data for all epochs and
    # energy bins. This will be a list of IntensityFactors named tuples for each epoch
    factors_per_epoch = [
        get_intensity_factors(
            species_energy_bins, species_ancillary_data_by_state[state]
        )
        for state in dynamic_threshold_states
    ]

    # Stack factors into arrays for vectorized computation
    delta_e = np.stack([factor.delta_e for factor in factors_per_epoch])
    geometry_factors = np.stack(
        [factor.geometry_factor for factor in factors_per_epoch]
    )
    efficiencies = np.stack([factor.efficiency for factor in factors_per_epoch])
    b = np.stack([factor.b for factor in factors_per_epoch])
    seconds = SECONDS_PER_MIN

    # Handle sectored rates which are multidimensional
    # (epoch, energy, azimuth, declination)
    if "declination" in updated_ds[species_variable].dims:
        # The factors are 1D arrays containing values for each declination (8) and each
        # energy bin. Reshape factors to match the dimensions of the sectored rates
        delta_e = delta_e.reshape((delta_e.shape[0], len(species_energy_bins), 8))[
            :, :, np.newaxis, :
        ]
        geometry_factors = geometry_factors.reshape(
            (geometry_factors.shape[0], len(species_energy_bins), 8)
        )[:, :, np.newaxis, :]
        efficiencies = efficiencies.reshape(
            (efficiencies.shape[0], len(species_energy_bins), 8)
        )[:, :, np.newaxis, :]
        b = b.reshape((b.shape[0], len(species_energy_bins), 8))[:, :, np.newaxis, :]
        seconds = SECONDS_PER_10_MIN

    # Store the factor arrays in a named tuple
    factors = IntensityFactors(
        delta_e=delta_e,
        geometry_factor=geometry_factors,
        efficiency=efficiencies,
        b=b,
        seconds=seconds,
    )

    # Calculate intensities using vectorized operations
    updated_ds[species_variable] = calculate_intensities(
        updated_ds[species_variable], factors
    )

    return updated_ds


def calculate_intensities_for_all_species(
    l2_dataset: xr.Dataset, ancillary_data_frames: dict, valid_data_variables: list
) -> xr.Dataset:
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
    valid_data_variables : list
        A list of valid data variables to calculate intensity for.

    Returns
    -------
    updated_ds : xr.Dataset
        The updated dataset with the intensity calculated for each species.
    """
    updated_ds = l2_dataset.copy()

    # Add statistical uncertainty variables to the list of valid variables
    data_variables = (
        valid_data_variables
        + [f"{var}_stat_uncert_delta_minus" for var in valid_data_variables]
        + [f"{var}_stat_uncert_delta_plus" for var in valid_data_variables]
    )

    # Calculate the intensity for each valid data variable
    for species_variable in data_variables:
        if species_variable in updated_ds.data_vars:
            updated_ds = calculate_intensities_for_a_species(
                species_variable, updated_ds, ancillary_data_frames
            )
        else:
            logger.warning(
                f"Variable {species_variable} not found in dataset. "
                f"Skipping intensity calculation."
            )

    return updated_ds


def add_systematic_uncertainties(
    dataset: xr.Dataset, particle: str, energy_bins: int
) -> xr.Dataset:
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

    Returns
    -------
    updated_ds : xr.Dataset
        The dataset with the systematic uncertainties added.
    """
    updated_ds = dataset.copy()

    updated_ds[f"{particle}_sys_err_minus"] = xr.DataArray(
        data=np.zeros(energy_bins, dtype=np.float32),
        dims=[f"{particle}_energy_mean"],
        name=f"{particle}_sys_err_minus",
    )
    updated_ds[f"{particle}_sys_err_plus"] = xr.DataArray(
        data=np.zeros(energy_bins, dtype=np.float32),
        dims=[f"{particle}_energy_mean"],
        name=f"{particle}_sys_err_plus",
    )

    return updated_ds


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

    # Remove possible trailing spaces from all values in the DataFrame
    ancillary_data = ancillary_data.map(
        lambda x: x.strip() if isinstance(x, str) else x
    )

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

    # Add systematic uncertainties to the dataset. These will not
    # have the intensity calculation applied to them
    for var in l2_summed_intensity_dataset.data_vars:
        if var in VALID_SPECIES:
            l2_summed_intensity_dataset = add_systematic_uncertainties(
                l2_summed_intensity_dataset,
                str(var),
                l2_summed_intensity_dataset[var].shape[1],
            )
    l2_summed_intensity_dataset = calculate_intensities_for_all_species(
        l2_summed_intensity_dataset, ancillary_data_frames, VALID_SPECIES
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

    Intensity is then calculated from the summed standard rates:

        Equation 9 from the HIT algorithm document:
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
        l2_standard_intensity_dataset = add_systematic_uncertainties(
            l2_standard_intensity_dataset, particle, len(energy_ranges)
        )
        # Add standard particle rates and statistical uncertainties to the dataset
        l2_standard_intensity_dataset = add_summed_particle_data_to_dataset(
            l2_standard_intensity_dataset,
            l1b_standard_rates_dataset,
            particle,
            energy_ranges,
        )
    l2_standard_intensity_dataset = calculate_intensities_for_all_species(
        l2_standard_intensity_dataset, ancillary_data_frames, VALID_SPECIES
    )

    return l2_standard_intensity_dataset


def process_sectored_intensity_data(
    l1b_sectored_rates_dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Will process L2 HIT sectored intensity data from L1B sectored rates data.

    This function converts the L1B sectored rates to L2 sectored intensities
    using ancillary tables containing factors needed to calculate the
    intensity (energy bin width, geometry factor, efficiency, and b).

    Equation 12 from the HIT algorithm document:
    Sectored Intensity = (Summed L1B Sectored Rates) /
                       (600 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    l1b_sectored_rates_dataset : xr.Dataset
        The L1B sectored rates dataset.

    Returns
    -------
    xr.Dataset
        The processed L2 sectored intensity dataset.
    """
    # TODO:
    #  - consider setting valid particles list in constants file or at top of this file

    # Create a new dataset to store the L2 sectored intensity data
    l2_sectored_intensity_dataset = l1b_sectored_rates_dataset.copy(deep=True)

    # Load ancillary data for each dynamic threshold state into a dictionary
    ancillary_data_frames = load_ancillary_data(
        set(l2_sectored_intensity_dataset["dynamic_threshold_state"].values),
        L2_SECTORED_ANCILLARY_PATH_PREFIX,
    )

    # Add systematic uncertainties to the dataset. These will not
    # have the intensity calculation applied to them
    for var in l2_sectored_intensity_dataset.data_vars:
        if var in VALID_SECTORED_SPECIES:
            l2_sectored_intensity_dataset = add_systematic_uncertainties(
                l2_sectored_intensity_dataset,
                str(var),
                l2_sectored_intensity_dataset[var].shape[1],
            )
    l2_sectored_intensity_dataset = calculate_intensities_for_all_species(
        l2_sectored_intensity_dataset, ancillary_data_frames, VALID_SECTORED_SPECIES
    )

    return l2_sectored_intensity_dataset
