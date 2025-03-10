"""IMAP-HIT L2 data processing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.hit_utils import get_attribute_manager

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
    #  Process standard rates dataset
    #  add logical sources for other l2 products
    #  "imap_hit_l2_standard-fluxes", "imap_hit_l2_sectored-fluxes"

    # Create L2 datasets
    l2_datasets: dict = {}

    if "imap_hit_l1b_summed-rates" in dependency.attrs["Logical_source"]:
        # Process science data to L2 datasets
        l2_datasets["imap_hit_l2_summed-fluxes"] = process_summed_flux_data(dependency)
        logger.info("HIT L2 summed flux dataset created")

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
