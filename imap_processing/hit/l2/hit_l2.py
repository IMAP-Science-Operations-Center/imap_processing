"""IMAP-HIT L2 data processing."""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.hit_utils import (
    add_energy_variables,
    get_attribute_manager,
    initialize_particle_data_arrays,
    sum_particle_data,
)
from imap_processing.hit.l2.constants import (
    PARTICLE_ENERGY_RANGE_MAPPING,
)

logger = logging.getLogger(__name__)

# TODO review logging levels to use (debug vs. info)


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
        List of L2 dataset.
    """
    logger.info("Creating HIT L2 science datasets")
    # Create the attribute manager for this data level
    attr_mgr = get_attribute_manager(data_version, "l2")

    # TODO: Write functions to create the following datasets
    #  Process sectored rates dataset
    #  add logical sources for other l2 products "imap_hit_l2_sectored-intensity"

    # Create L2 datasets
    l2_datasets: dict = {}

    # Process science data to L2 datasets
    if "imap_hit_l1b_summed-rates" in dependency.attrs["Logical_source"]:
        l2_datasets["imap_hit_l2_summed-intensity"] = process_summed_flux_data(
            dependency
        )
        logger.info("HIT L2 summed intensity dataset created")

    if "imap_hit_l1b_standard-rates" in dependency.attrs["Logical_source"]:
        l2_datasets["imap_hit_l2_standard-intensity"] = process_summed_flux_data(
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


def process_summed_flux_data(l1b_summed_rates_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process L2 HIT summed flux data from L1B summed rates.

    This function converts the L1B summed rates to L2 summed fluxes
    using ancillary tables containing factors needed to calculate the
    flux (energy bin width, geometry factor, efficiency, and b).

    Equation 11 from the HIT algorithm document:
      Summed Flux = (L1B Summed Rate) /
                    (60 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    l1b_summed_rates_dataset : xarray.Dataset
        HIT L1B summed rates dataset.

    Returns
    -------
    xr.Dataset
        The processed L2 summed flux dataset.
    """
    # TODO:
    #  - determine where to pull ancillary data. Storing it locally for now
    #  - add check for dynamic_threshold_state to determine which ancillary table to use
    #    after additional ancillary files are provided

    # Create a new dataset to store the L1B summed flux data
    l2_summed_flux_dataset = l1b_summed_rates_dataset.copy(deep=True)

    # Load ancillary data containing factors needed to convert rate to flux
    # (energy bin width, geometry factor, efficiency, and b)
    ancillary_file = (
        imap_module_directory
        / "hit/ancillary/imap_hit_l1b-to-l2-summed-factors-20250219_v002.csv"
    )
    ancillary_data = pd.read_csv(ancillary_file)

    # Convert column names and species values to lowercase
    ancillary_data.columns = ancillary_data.columns.str.lower().str.strip()
    ancillary_data["species"] = ancillary_data["species"].str.lower()

    # Calculate the summed flux using the appropriate ancillary table.
    for var in l2_summed_flux_dataset.data_vars:
        if var != "dynamic_threshold_state" and "energy_" not in var:
            # Get the species name from the variable name
            species = str(var).split("_")[0] if "_delta_" in var else var

            # Get the ancillary data for the species
            var_anc_data = ancillary_data[ancillary_data["species"] == species]

            # Calculate the summed flux for each epoch and energy bin
            for epoch in range(l2_summed_flux_dataset[var].shape[0]):
                # TODO: Add check for energy max after updated ancillary file is
                #  provided fixing errors
                # Get the energy min values for the current epoch
                energy_min = l2_summed_flux_dataset[f"{species}_energy_min"].values

                # Get factors needed to convert summed rates to fluxes for
                # all energy bins
                flux_factors = var_anc_data.set_index(
                    var_anc_data["lower energy (mev)"].astype(np.float32)
                ).loc[energy_min]
                delta_e_factor = flux_factors["delta e (mev)"].values
                geometry_factor = flux_factors["geometry factor (cm2 sr)"].values
                efficiency = flux_factors["efficiency"].values
                b = flux_factors["b"].values

                # Calculate the summed flux for this energy bin
                l2_summed_flux_dataset[var][epoch] = (
                    l2_summed_flux_dataset[var][epoch]
                    / (60 * delta_e_factor * geometry_factor * efficiency)
                ) - b
    return l2_summed_flux_dataset


class SummedRates(NamedTuple):
    """A namedtuple to store summed rates and uncertainties."""

    summed_rates: xr.DataArray
    summed_rates_delta_minus: xr.DataArray
    summed_rates_delta_plus: xr.DataArray


def calculate_flux(l2_dataset: xr.Dataset, ancillary_data_frames: dict) -> None:
    """
    Calculate the flux for each species in the dataset.

    This function uses equation 9 and 11 from the HIT algorithm document:
        (Summed L1B Rates) / (60 * Delta E * Geometry Factor * Efficiency) - b

    The summed L1B rates are in the l2 dataset passed in.

    Parameters
    ----------
    l2_dataset : xr.Dataset
        The L2 dataset to calculate the flux for.
    ancillary_data_frames : dict
        Dictionary containing ancillary data for each dynamic threshold state.
    """
    for var in l2_dataset.data_vars:
        if (
            var != "dynamic_threshold_state"
            and "energy_" not in var
            and "sys" not in var
        ):
            # Get the species name from the variable name
            species = str(var).split("_")[0] if "_delta_" in var else var

            # Calculate the summed flux for each epoch and energy bin
            for epoch in range(l2_dataset[var].shape[0]):
                # TODO: Add check for energy max after ancillary file is updated
                #  to fix errors
                # Get the energy min values for the current epoch
                energy_min = (
                    l2_dataset[f"{species}_energy_mean"].values
                    - l2_dataset[f"{species}_energy_delta_minus"].values
                )

                # Get the correct ancillary data using the dynamic threshold state
                dynamic_threshold_state = int(
                    l2_dataset["dynamic_threshold_state"][epoch].values
                )
                ancillary_data = ancillary_data_frames[dynamic_threshold_state]

                # Get the ancillary data for the species
                var_anc_data = ancillary_data[ancillary_data["species"] == species]

                # Get factors needed to convert summed rates to fluxes for
                # all energy bins for the species
                flux_factors = var_anc_data.set_index(
                    var_anc_data["lower energy (mev)"].astype(np.float32)
                ).loc[energy_min]
                delta_e_factor = flux_factors["delta e (mev)"].values
                geometry_factor = flux_factors["geometry factor (cm2 sr)"].values
                efficiency = flux_factors["efficiency"].values
                b = flux_factors["b"].values

                # Calculate the summed flux for this energy bin
                l2_dataset[var][epoch] = (
                    l2_dataset[var][epoch]
                    / (60 * delta_e_factor * geometry_factor * efficiency)
                ) - b


def add_summed_particle_rates(
    l2_standard_flux_dataset: xr.Dataset,
    l1b_standard_rates_dataset: xr.Dataset,
    particle_energy_range_mapping: dict,
) -> None:
    """
    Add summed particle rates to the dataset.

    This function adds the summed rates from the l2fgrates, l3fgrates, and penfgrates
    data variables in the L1B standard rates data to the L2 standard flux dataset by
    particle type and energy range.

    Parameters
    ----------
    l2_standard_flux_dataset : xr.Dataset
        The L2 standard flux dataset to add the summed rates to.
    l1b_standard_rates_dataset : xr.Dataset
        The L1B standard rates dataset containing rates to sum.
    particle_energy_range_mapping : dict
        Dictionary mapping particles to their energy and detector ranges.
    """
    for particle, energy_ranges in particle_energy_range_mapping.items():
        # Initialize arrays to store summed rates and statistical uncertainties
        l2_standard_flux_dataset = initialize_particle_data_arrays(
            l2_standard_flux_dataset,
            particle,
            len(energy_ranges),
            l1b_standard_rates_dataset.sizes["epoch"],
        )

        # initialize arrays to store energy min and max values
        energy_min = np.zeros(len(energy_ranges), dtype=np.float32)
        energy_max = np.zeros(len(energy_ranges), dtype=np.float32)

        # Sum particle rates for each energy range and add them to the dataset
        for i, energy_range in enumerate(energy_ranges):
            summed_rates, summed_rates_delta_minus, summed_rates_delta_plus = (
                sum_particle_data(l1b_standard_rates_dataset, energy_range)
            )

            # Create namedtuple to store summed counts and uncertainties
            summed_rates = SummedRates(
                summed_rates, summed_rates_delta_minus, summed_rates_delta_plus
            )

            # Add summed rates to the dataset
            l2_standard_flux_dataset[f"{particle}"][:, i] = (
                summed_rates.summed_rates.astype(np.float32)
            )
            l2_standard_flux_dataset[f"{particle}_delta_minus"][:, i] = (
                summed_rates.summed_rates_delta_minus.astype(np.float32)
            )
            l2_standard_flux_dataset[f"{particle}_delta_plus"][:, i] = (
                summed_rates.summed_rates_delta_plus.astype(np.float32)
            )

            # Add systematic uncertainties to the dataset. Just zeros for now.
            # To change if/when HIT determines there are systematic uncertainties
            l2_standard_flux_dataset[f"{particle}_sys_delta_minus"] = xr.DataArray(
                data=np.zeros(len(energy_ranges), dtype=np.float32),
                dims=[f"{particle}_energy_mean"],
                name=f"{particle}_sys_delta_minus",
            )
            l2_standard_flux_dataset[f"{particle}_sys_delta_plus"] = xr.DataArray(
                data=np.zeros(len(energy_ranges), dtype=np.float32),
                dims=[f"{particle}_energy_mean"],
                name=f"{particle}_sys_delta_plus",
            )

            # Fill energy min and max values for each energy range
            energy_min[i] = energy_range["energy_min"]
            energy_max[i] = energy_range["energy_max"]

        l2_standard_flux_dataset = add_energy_variables(
            l2_standard_flux_dataset, particle, energy_min, energy_max
        )


def process_standard_flux_data(l1b_standard_rates_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process L2 standard flux data from L1B standard rates data.

    This function converts the L1B standard rates to L2 standard fluxes
    for each particle type and energy range using ancillary tables containing
    factors needed to calculate the flux (energy bin width, geometry factor,
    efficiency, and b).

    First, rates from the l2fgrates, l3fgrates, and penfgrates
    data variables in the L1B standard rates data are summed.
    These variables represent rates for different detector penetration ranges
    (Range 2, Range 3, and Range 4 respectively). Only the energy ranges specified
    in the PARTICLE_ENERGY_RANGE_MAPPING dictionary are included in this product.

    Flux is then calculated from the summed rates using the following equation:

        Equation 9 from the HIT algorithm document:
            Standard Flux = (Summed L1B Standard Rates) /
                            (60 * Delta E * Geometry Factor * Efficiency) - b

    Parameters
    ----------
    l1b_standard_rates_dataset : xr.Dataset
        The L1B standard rates dataset.

    Returns
    -------
    xr.Dataset
        The processed L1B summed rates dataset.
    """
    # Create a new dataset to store the L2 standard flux data
    l2_standard_flux_dataset = xr.Dataset()

    # Add dynamic threshold state to the dataset to use with ancillary data
    l2_standard_flux_dataset["dynamic_threshold_state"] = l1b_standard_rates_dataset[
        "dynamic_threshold_state"
    ]

    # Assign the epoch coordinate from the l1B dataset
    l2_standard_flux_dataset = l2_standard_flux_dataset.assign_coords(
        {"epoch": l1b_standard_rates_dataset.coords["epoch"]}
    )

    # Load ancillary data containing factors needed to convert rates to flux.
    # Which ancillary file to use depends on the dynamic threshold state (0-3).
    # Build a dictionary with ancillary data for each dynamic threshold state
    # in the dataset.
    path_prefix = imap_module_directory / "hit/ancillary/imap_hit_l1b-to-l2-standard-dt"
    ancillary_data_frames = {
        int(state): pd.read_csv(f"{path_prefix}{state}-factors_20250219_v002.csv")
        for state in set(l2_standard_flux_dataset["dynamic_threshold_state"].values)
    }

    # Convert column names and species values to lowercase
    for df in ancillary_data_frames.values():
        df.columns = df.columns.str.lower().str.strip()
        df["species"] = df["species"].str.lower()

    add_summed_particle_rates(
        l2_standard_flux_dataset,
        l1b_standard_rates_dataset,
        PARTICLE_ENERGY_RANGE_MAPPING,
    )
    calculate_flux(l2_standard_flux_dataset, ancillary_data_frames)
    l2_standard_flux_dataset = l2_standard_flux_dataset.drop_vars(
        "dynamic_threshold_state"
    )

    return l2_standard_flux_dataset
