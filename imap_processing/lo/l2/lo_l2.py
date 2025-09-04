"""IMAP-Lo L2 data processing."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.ena_maps import AbstractSkyMap, RectangularSkyMap
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo import lo_ancillary
from imap_processing.spice.time import et_to_datetime64, ttj2000ns_to_et

logger = logging.getLogger(__name__)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def lo_l2(
    sci_dependencies: dict, anc_dependencies: list, descriptor: str
) -> list[xr.Dataset]:
    """
    Process IMAP-Lo L1C data into L2 CDF data products.

    This is the main entry point for L2 processing. It orchestrates the entire
    processing pipeline from L1C pointing sets to L2 sky maps with intensities.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L2 data product creation in xarray Datasets.
        Must contain "imap_lo_l1c_pset" key with list of pointing set datasets.
    anc_dependencies : list
        List of ancillary file paths needed for L2 data product creation.
        Should include efficiency factor files.
    descriptor : str
        The map descriptor to be produced
        (e.g., "ilo90-ena-h-sf-nsp-full-hae-6deg-3mo").

    Returns
    -------
    list[xr.Dataset]
        List containing the processed L2 dataset with rates, intensities,
        and uncertainties.

    Raises
    ------
    ValueError
        If no pointing set data found in science dependencies.
    NotImplementedError
        If HEALPix map output is requested (only rectangular maps supported).
    """
    logger.info("Starting IMAP-Lo L2 processing pipeline")
    if "imap_lo_l1c_pset" not in sci_dependencies:
        raise ValueError("No pointing set data found in science dependencies")
    psets = sci_dependencies["imap_lo_l1c_pset"]

    # TODO: Remove this hardcoded logical source
    logical_source = "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-3mo"

    logger.info("Step 1: Loading ancillary data")
    efficiency_data = load_efficiency_data(anc_dependencies)

    logger.info(f"Step 2: Creating sky map from {len(psets)} pointing sets")
    sky_map = create_sky_map_from_psets(psets, descriptor, efficiency_data)

    logger.info("Step 3: Converting to dataset and adding geometric factors")
    dataset = sky_map.to_dataset()
    dataset = add_geometric_factors(dataset)

    logger.info("Step 4: Calculating rates and intensities")
    dataset = calculate_all_rates_and_intensities(dataset)

    logger.info("Step 5: Finalizing dataset with attributes")
    dataset = finalize_dataset(dataset, logical_source)

    logger.info("IMAP-Lo L2 processing pipeline completed successfully")
    return [dataset]


# =============================================================================
# SETUP AND INITIALIZATION HELPERS
# =============================================================================


def load_efficiency_data(anc_dependencies: list) -> pd.DataFrame:
    """
    Load efficiency factor data from ancillary files.

    Parameters
    ----------
    anc_dependencies : list
        List of ancillary file paths to search for efficiency factor files.

    Returns
    -------
    pd.DataFrame
        Concatenated efficiency factor data from all matching files.
        Returns empty DataFrame if no efficiency files found.
    """
    efficiency_files = [
        anc_file
        for anc_file in anc_dependencies
        if "efficiency-factor" in str(anc_file)
    ]

    if not efficiency_files:
        logger.warning("No efficiency factor files found in ancillary dependencies")
        return pd.DataFrame()

    logger.debug(f"Loading {len(efficiency_files)} efficiency factor files")
    return pd.concat(
        [lo_ancillary.read_ancillary_file(anc_file) for anc_file in efficiency_files],
        ignore_index=True,
    )


def finalize_dataset(dataset: xr.Dataset, logical_source: str) -> xr.Dataset:
    """
    Add attributes and perform final dataset preparation.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to finalize with attributes.
    logical_source : str
        The logical source identifier for global attributes.

    Returns
    -------
    xr.Dataset
        The finalized dataset with all attributes added.
    """
    # Initialize the attribute manager
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-common")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-rectangular")

    # Add global and variable attributes
    dataset.attrs.update(attr_mgr.get_global_attributes(logical_source))
    for var in dataset.data_vars:
        try:
            dataset[var].attrs = attr_mgr.get_variable_attributes(var)
        except KeyError:
            # If no attributes found, try without schema validation
            try:
                dataset[var].attrs = attr_mgr.get_variable_attributes(
                    var, check_schema=False
                )
            except KeyError:
                logger.warning(f"No attributes found for variable {var}")

    return dataset


# =============================================================================
# SKY MAP CREATION PIPELINE
# =============================================================================


def create_sky_map_from_psets(
    psets: list[xr.Dataset], descriptor: str, efficiency_data: pd.DataFrame
) -> AbstractSkyMap:
    """
    Create a sky map by processing all pointing sets.

    Parameters
    ----------
    psets : list[xr.Dataset]
        List of pointing set datasets to process.
    descriptor : str
        Map descriptor string defining the projection and binning.
    efficiency_data : pd.DataFrame
        Efficiency factor data for correcting counts.

    Returns
    -------
    AbstractSkyMap
        The populated sky map with projected data from all pointing sets.

    Raises
    ------
    NotImplementedError
        If HEALPix map output is requested (only rectangular maps supported).
    """
    # Initialize the output map
    map_descriptor = MapDescriptor.from_string(descriptor)
    output_map = map_descriptor.to_empty_map()

    if not isinstance(output_map, RectangularSkyMap):
        raise NotImplementedError("HEALPix map output not supported for Lo")

    logger.debug(f"Processing {len(psets)} pointing sets")
    # Process each pointing set
    for i, pset in enumerate(psets):
        logger.debug(f"Processing pointing set {i + 1}/{len(psets)}")
        processed_pset = process_single_pset(pset, output_map, efficiency_data)
        project_pset_to_map(processed_pset, output_map)

    return output_map


def process_single_pset(
    pset: xr.Dataset, output_map: AbstractSkyMap, efficiency_data: pd.DataFrame
) -> xr.Dataset:
    """
    Process a single pointing set for projection to the sky map.

    Parameters
    ----------
    pset : xr.Dataset
        Single pointing set dataset to process.
    output_map : AbstractSkyMap
        The target sky map for coordinate alignment.
    efficiency_data : pd.DataFrame
        Efficiency factor data for correcting counts.

    Returns
    -------
    xr.Dataset
        Processed pointing set ready for projection with efficiency corrections applied.
    """
    # Step 1: Normalize coordinate system
    pset_processed = normalize_pset_coordinates(pset, output_map)

    # Step 2: Add efficiency factors
    pset_processed = add_efficiency_factors_to_pset(pset_processed, efficiency_data)

    # Step 3: Calculate efficiency-corrected quantities
    pset_processed = calculate_efficiency_corrected_quantities(pset_processed)

    return pset_processed


def normalize_pset_coordinates(
    pset: xr.Dataset, output_map: AbstractSkyMap
) -> xr.Dataset:
    """
    Normalize pointing set coordinates to match the output map.

    Parameters
    ----------
    pset : xr.Dataset
        Input pointing set dataset with potentially mismatched coordinates.
    output_map : AbstractSkyMap
        Target sky map for coordinate alignment.

    Returns
    -------
    xr.Dataset
        Pointing set with normalized energy coordinates and dimension names.
    """
    # Ensure consistent energy coordinates (maps want energy not esa_energy_step)
    pset_renamed = pset.rename_dims({"esa_energy_step": "energy"})

    # Drop the esa_energy_step coordinate first to avoid conflicts
    if "esa_energy_step" in pset_renamed.variables:
        pset_renamed = pset_renamed.drop_vars("esa_energy_step")

    # Ensure the pset energy coordinates match the output map
    if "energy" in output_map.data_1d.dims:
        # Get the energy coordinates from the output map
        map_energy_coords = output_map.data_1d.coords.get("energy", range(7))
        # Align the pset energy coordinates to match the map
        pset_renamed = pset_renamed.assign_coords(energy=map_energy_coords)

    return pset_renamed


def add_efficiency_factors_to_pset(
    pset: xr.Dataset, efficiency_data: pd.DataFrame
) -> xr.Dataset:
    """
    Add efficiency factors to the pointing set based on observation date.

    Parameters
    ----------
    pset : xr.Dataset
        Pointing set dataset to add efficiency factors to.
    efficiency_data : pd.DataFrame
        Efficiency factor data containing date-indexed efficiency values.

    Returns
    -------
    xr.Dataset
        Pointing set with efficiency factors added as new data variable.

    Raises
    ------
    ValueError
        If no efficiency factor found for the pointing set observation date.
    """
    if efficiency_data.empty:
        # If no efficiency data, create unity efficiency
        logger.warning("No efficiency data available, using unity efficiency")
        pset["efficiency"] = xr.DataArray(np.ones(7), dims=["energy"])
        return pset

    # Convert the epoch to datetime64
    date = et_to_datetime64(ttj2000ns_to_et(pset["epoch"].values[0]))
    # The efficiency file only has date as YYYYDDD, so drop the time for this
    date = date.astype("M8[D]")  # Convert to date only (no time)

    ef_df = efficiency_data[efficiency_data["Date"] == date]
    if ef_df.empty:
        raise ValueError(f"No efficiency factor found for pset date {date}")

    efficiency_values = ef_df[
        [
            "E-Step1_eff",
            "E-Step2_eff",
            "E-Step3_eff",
            "E-Step4_eff",
            "E-Step5_eff",
            "E-Step6_eff",
            "E-Step7_eff",
        ]
    ].values[0]

    pset["efficiency"] = xr.DataArray(
        efficiency_values,
        dims=["energy"],
    )
    logger.debug(f"Applied efficiency factors for date {date}")
    return pset


def calculate_efficiency_corrected_quantities(pset: xr.Dataset) -> xr.Dataset:
    """
    Calculate efficiency-corrected quantities for each particle type.

    Parameters
    ----------
    pset : xr.Dataset
        Pointing set with efficiency factors applied.

    Returns
    -------
    xr.Dataset
        Pointing set with efficiency-corrected count variables added.
    """
    for var in ["h", "o", "doubles", "triples"]:
        # counts / efficiency
        pset[f"{var}_counts_over_eff"] = pset[f"{var}_counts"] / pset["efficiency"]
        # counts / efficiency**2 (for variance propagation)
        pset[f"{var}_counts_over_eff_squared"] = pset[f"{var}_counts"] / (
            pset["efficiency"] ** 2
        )

    return pset


def project_pset_to_map(pset: xr.Dataset, output_map: AbstractSkyMap) -> None:
    """
    Project pointing set data to the output map.

    Parameters
    ----------
    pset : xr.Dataset
        Processed pointing set ready for projection.
    output_map : AbstractSkyMap
        Target sky map to receive the projected data.

    Returns
    -------
    None
        Function modifies output_map in place.
    """
    # Define base quantities to project
    value_keys = ["exposure_time"]

    # Add quantities for each particle type that exists in the dataset
    for var in ["h", "o", "doubles", "triples"]:
        if f"{var}_counts" in pset.data_vars:
            value_keys.extend(
                [
                    f"{var}_counts",
                    f"{var}_counts_over_eff",
                    f"{var}_counts_over_eff_squared",
                ]
            )

    # Create LoPointingSet and project to map
    lo_pset = ena_maps.LoPointingSet(pset)
    output_map.project_pset_values_to_map(
        pointing_set=lo_pset,
        value_keys=value_keys,
        index_match_method=ena_maps.IndexMatchMethod.PUSH,
    )
    logger.debug(f"Projected {len(value_keys)} quantities to sky map")


# =============================================================================
# GEOMETRIC FACTORS
# =============================================================================


def add_geometric_factors(dataset: xr.Dataset) -> xr.Dataset:
    """
    Add geometric factors to the sky map after projection.

    Parameters
    ----------
    dataset : xr.Dataset
        Sky map dataset to add geometric factors to.

    Returns
    -------
    xr.Dataset
        Dataset with geometric factor variables added for each energy step.
    """
    logger.info("Loading and applying geometric factors")
    # Load geometric factor data
    h_gf_data, o_gf_data = load_geometric_factor_data()

    # Initialize geometric factor variables
    dataset = initialize_geometric_factor_variables(dataset)

    # Populate geometric factors for each energy step
    dataset = populate_geometric_factors(dataset, h_gf_data, o_gf_data)

    return dataset


def load_geometric_factor_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load hydrogen and oxygen geometric factor data from ancillary files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Hydrogen and oxygen geometric factor dataframes.
    """
    anc_path = Path(__file__).parent.parent / "ancillary_data"

    h_gf_df = lo_ancillary.read_ancillary_file(
        anc_path / "imap_lo_hydrogen-geometric-factor_v001.csv"
    )
    o_gf_df = lo_ancillary.read_ancillary_file(
        anc_path / "imap_lo_oxygen-geometric-factor_v001.csv"
    )

    return h_gf_df, o_gf_df


def initialize_geometric_factor_variables(dataset: xr.Dataset) -> xr.Dataset:
    """
    Initialize all geometric factor variables with proper dimensions.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset to add geometric factor variables to.

    Returns
    -------
    xr.Dataset
        Dataset with initialized geometric factor variables for all energy steps.
    """
    gf_vars = [
        "energy_h",
        "energy_h_stat_uncert",
        "h_gf",
        "h_gf_stat_uncert",
        "energy_o",
        "energy_o_stat_uncert",
        "o_gf",
        "o_gf_stat_uncert",
        "doubles_gf",
        "doubles_gf_stat_uncert",
        "triples_gf",
        "triples_gf_stat_uncert",
    ]

    # Initialize all variables with proper dimensions (energy only)
    for var in gf_vars:
        dataset[var] = xr.DataArray(
            np.zeros(7),
            dims=["energy"],
        )

    return dataset


def populate_geometric_factors(
    dataset: xr.Dataset, h_gf_data: pd.DataFrame, o_gf_data: pd.DataFrame
) -> xr.Dataset:
    """
    Populate geometric factor values for each energy step.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with initialized geometric factor variables.
    h_gf_data : pd.DataFrame
        Hydrogen geometric factor data from ancillary files.
    o_gf_data : pd.DataFrame
        Oxygen geometric factor data from ancillary files.

    Returns
    -------
    xr.Dataset
        Dataset with populated geometric factor values for all energy steps.
    """
    # Mapping of dataset variables to dataframe columns
    gf_vars = {
        "energy_h": "Cntr_E",
        "energy_h_stat_uncert": "Cntr_E_unc",
        "h_gf": "GF_Trpl_H",
        "h_gf_stat_uncert": "GF_Trpl_H_unc",
        "energy_o": "Cntr_E",
        "energy_o_stat_uncert": "Cntr_E_unc",
        "o_gf": "GF_Trpl_O",
        "o_gf_stat_uncert": "GF_Trpl_O_unc",
        "doubles_gf": "GF_Dbl_all",
        "doubles_gf_stat_uncert": "GF_Dbl_all_unc",
        "triples_gf": "GF_Trpl_all",
        "triples_gf_stat_uncert": "GF_Trpl_all_unc",
    }

    # Get ESA mode from the map (assuming it's constant or we take the first)
    # TODO: Figure out how to handle esa_mode properly
    if "esa_mode" in dataset:
        esa_mode = dataset["esa_mode"].values[0]
    else:
        # Default to mode 0 if not available (HiRes mode)
        esa_mode = 0

    # Populate the geometric factors for each energy step
    for i in range(7):
        # Get geometric factor data for this energy step and ESA mode
        h_gf_row = h_gf_data[
            (h_gf_data["esa_mode"] == esa_mode)
            & (h_gf_data["Observed_E-Step"] == i + 1)
        ].iloc[0]
        o_gf_row = o_gf_data[
            (o_gf_data["esa_mode"] == esa_mode)
            & (o_gf_data["Observed_E-Step"] == i + 1)
        ].iloc[0]

        # Fill energy step with the geometric factor values
        for var, col in gf_vars.items():
            if var.startswith("energy_h") or var.startswith("h_gf"):
                dataset[var].values[i] = h_gf_row[col]
            elif var.startswith("energy_o") or var.startswith("o_gf"):
                dataset[var].values[i] = o_gf_row[col]
            elif var.endswith("_gf"):
                # These are general geometric factors from hydrogen file
                dataset[var].values[i] = h_gf_row[col]

    return dataset


# =============================================================================
# RATES AND INTENSITIES CALCULATIONS
# =============================================================================


def calculate_all_rates_and_intensities(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate rates and intensities with proper error propagation.

    Parameters
    ----------
    dataset : xr.Dataset
        Sky map dataset with count data and geometric factors.

    Returns
    -------
    xr.Dataset
        Dataset with calculated rates, intensities, and uncertainties for all
        particle types.
    """
    # Step 1: Calculate rates for all particle types
    dataset = calculate_rates(dataset)

    # Step 2: Calculate intensities for H and O only
    dataset = calculate_intensities(dataset)

    # Step 3: Clean up intermediate variables
    dataset = cleanup_intermediate_variables(dataset)

    return dataset


def calculate_rates(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate count rates and their statistical uncertainties.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with count data and exposure times.

    Returns
    -------
    xr.Dataset
        Dataset with calculated count rates and statistical uncertainties
        for all particle types.
    """
    for var in ["h", "o", "doubles", "triples"]:
        # Rate = counts / exposure_time
        dataset[f"{var}_rate"] = dataset[f"{var}_counts"] / dataset["exposure_time"]

        # Poisson uncertainty on the counts propagated to the rate
        # TODO: Is there uncertainty in the exposure time too?
        dataset[f"{var}_rate_stat_uncert"] = (
            np.sqrt(dataset[f"{var}_counts"]) / dataset["exposure_time"]
        )

    return dataset


def calculate_intensities(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate particle intensities and uncertainties for H and O.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with count rates, geometric factors, and center energies.

    Returns
    -------
    xr.Dataset
        Dataset with calculated particle intensities and their statistical
        and systematic uncertainties for hydrogen and oxygen.
    """
    for var in ["h", "o"]:
        # Equation 3 from mapping document (average intensity)
        dataset[f"{var}_intensity"] = dataset[f"{var}_counts_over_eff"] / (
            dataset[f"{var}_gf"] * dataset[f"energy_{var}"] * dataset["exposure_time"]
        )

        # Equation 4 from mapping document (statistical uncertainty)
        # Note that we need to take the square root to get the uncertainty as
        # the equation is for the variance
        dataset[f"{var}_intensity_stat_uncert"] = np.sqrt(
            dataset[f"{var}_counts_over_eff_squared"]
            / (
                dataset[f"{var}_gf"]
                * dataset[f"energy_{var}"]
                * dataset["exposure_time"]
            )
        )

        # Equation 5 from mapping document (systematic uncertainty)
        dataset[f"{var}_intensity_sys_err"] = (
            dataset[f"{var}_gf_stat_uncert"]
            / dataset[f"{var}_gf"]
            * dataset[f"{var}_intensity"]
        )  # TODO: Add background rates (only for H and O)
        # TODO: Add background intensities (only for H and O)

    return dataset


def cleanup_intermediate_variables(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remove intermediate variables that were only needed for calculations.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing intermediate calculation variables.

    Returns
    -------
    xr.Dataset
        Cleaned dataset with intermediate variables removed.
    """
    # Remove the intermediate variables from the map
    # i.e. the ones that were projected from the pset only for the purposes
    # of math and not desired in the output.
    vars_to_remove = []
    for var in ["h", "o", "doubles", "triples"]:
        # Only remove variables that exist in the dataset
        potential_vars = [
            f"{var}_counts_over_eff",
            f"{var}_counts_over_eff_squared",
            f"{var}_gf",
            f"{var}_gf_stat_uncert",
        ]
        for potential_var in potential_vars:
            if potential_var in dataset.data_vars:
                vars_to_remove.append(potential_var)

    return dataset.drop_vars(vars_to_remove)
