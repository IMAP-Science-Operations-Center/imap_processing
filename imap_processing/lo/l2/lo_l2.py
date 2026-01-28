"""IMAP-Lo L2 data processing."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.ena_maps import AbstractSkyMap, RectangularSkyMap
from imap_processing.ena_maps.utils.corrections import (
    PowerLawFluxCorrector,
    add_spacecraft_velocity_to_pset,
    apply_compton_getting_correction,
    calculate_ram_mask,
    get_pset_directional_mask,
    interpolate_map_flux_to_helio_frame,
)
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

    # Parse the map descriptor to get species and other attributes
    map_descriptor = MapDescriptor.from_string(descriptor)
    logger.info(f"Processing map for species: {map_descriptor.species}")

    # Determine if corrections are needed and prepare oxygen data if required
    (
        sputtering_correction,
        bootstrap_correction,
        flux_correction,
        o_map_dataset,
        flux_factors,
        cg_correction,
    ) = _prepare_corrections(
        map_descriptor, descriptor, sci_dependencies, anc_dependencies
    )

    logger.info("Step 1: Loading ancillary data")
    efficiency_data = load_efficiency_data(anc_dependencies)

    logger.info(f"Step 2: Creating sky map from {len(psets)} pointing sets")
    sky_map = create_sky_map_from_psets(
        psets, map_descriptor, efficiency_data, cg_correction
    )

    logger.info("Step 3: Converting to dataset and adding geometric factors")
    dataset = sky_map.to_dataset()
    dataset = add_geometric_factors(dataset, map_descriptor.species)

    logger.info("Step 4: Calculating rates and intensities")
    dataset = calculate_all_rates_and_intensities(
        dataset,
        sputtering_correction=sputtering_correction,
        bootstrap_correction=bootstrap_correction,
        flux_correction=flux_correction,
        o_map_dataset=o_map_dataset,
        flux_factors=flux_factors,
        cg_correction=cg_correction,
    )

    logger.info("Step 5: Finalizing dataset with attributes")
    dataset = sky_map.build_cdf_dataset(  # type: ignore[attr-defined]
        instrument="lo", level="l2", descriptor=descriptor, external_map_dataset=dataset
    )

    logger.info("IMAP-Lo L2 processing pipeline completed successfully")
    return [dataset]


def _prepare_corrections(
    map_descriptor: MapDescriptor,
    descriptor: str,
    sci_dependencies: dict,
    anc_dependencies: list,
) -> tuple[bool, bool, bool, xr.Dataset | None, Path | None, bool]:
    """
    Determine what corrections are needed and prepare oxygen dataset if required.

    This helper function encapsulates the logic for determining when sputtering
    and bootstrap corrections should be applied, and handles the creation of
    the oxygen dataset needed for sputtering corrections.

    Parameters
    ----------
    map_descriptor : MapDescriptor
        The parsed map descriptor containing species and data type information.
    descriptor : str
        The original descriptor string for creating the oxygen variant.
    sci_dependencies : dict
        Dictionary of datasets needed for L2 data product creation.
    anc_dependencies : list
        List of ancillary file paths.

    Returns
    -------
    tuple[bool, bool, bool, xr.Dataset | None, Path | None, bool]
        A tuple containing:
        - sputtering_correction: Whether to apply sputtering corrections
        - bootstrap_correction: Whether to apply bootstrap corrections
        - flux_correction: Whether to apply flux corrections
        - o_map_dataset: Oxygen dataset if needed, None otherwise
        - flux_factors: Path to flux factors ancillary file if needed,
         None otherwise
        - cg_correction: Whether to apply CG correction to the dataset.
    """
    # Default values - no corrections needed
    sputtering_correction = False
    bootstrap_correction = False
    flux_correction = False
    o_map_dataset = None
    flux_factors: None | Path = None

    # Sputtering and bootstrap corrections are only applied to hydrogen ENA data
    # Guard against recursion: don't process oxygen for oxygen maps
    if (
        map_descriptor.species == "h"
        and map_descriptor.principal_data == "ena"
        and "-o-" not in descriptor
    ):  # Safety check to prevent infinite recursion
        logger.info("Creating map for oxygen for sputtering corrections")
        o_descriptor = descriptor.replace("-h-", "-o-")
        o_map_dataset = lo_l2(sci_dependencies, anc_dependencies, o_descriptor)[0]
        sputtering_correction = True
        bootstrap_correction = True

    if "raw" not in map_descriptor.principal_data:
        flux_correction = True
        try:
            flux_factors = next(
                x for x in anc_dependencies if "esa-eta-fit-factors" in str(x)
            )
        except StopIteration:
            raise ValueError(
                "No flux correction factor file found in ancillary dependencies"
            ) from None

    cg_correction = True if map_descriptor.frame_descriptor == "hf" else False

    return (
        sputtering_correction,
        bootstrap_correction,
        flux_correction,
        o_map_dataset,
        flux_factors,
        cg_correction,
    )


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


def finalize_dataset(dataset: xr.Dataset, descriptor: str) -> xr.Dataset:
    """
    Add attributes and perform final dataset preparation.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to finalize with attributes.
    descriptor : str
        The descriptor for this map dataset.

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
    dataset.attrs.update(attr_mgr.get_global_attributes("imap_lo_l2_enamap"))

    # Our global attributes have placeholders for descriptor
    # so iterate through here and fill that in with the map-specific descriptor
    for key in ["Data_type", "Logical_source", "Logical_source_description"]:
        dataset.attrs[key] = dataset.attrs[key].format(descriptor=descriptor)
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
    psets: list[xr.Dataset],
    map_descriptor: MapDescriptor,
    efficiency_data: pd.DataFrame,
    cg_correct: bool,
) -> AbstractSkyMap:
    """
    Create a sky map by processing all pointing sets.

    Parameters
    ----------
    psets : list[xr.Dataset]
        List of pointing set datasets to process.
    map_descriptor : MapDescriptor
        Map descriptor object defining the projection and binning.
    efficiency_data : pd.DataFrame
        Efficiency factor data for correcting counts.
    cg_correct : bool
        Whether to apply the CG correction to each PSET.

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
    output_map = map_descriptor.to_empty_map()

    if not isinstance(output_map, RectangularSkyMap):
        raise NotImplementedError("HEALPix map output not supported for Lo")

    logger.debug(f"Processing {len(psets)} pointing sets")
    # Process each pointing set
    for i, pset in enumerate(psets):
        logger.debug(f"Processing pointing set {i + 1}/{len(psets)}")
        processed_pset = process_single_pset(
            pset,
            efficiency_data,
            map_descriptor.species,
            cg_correct,
        )
        directional_mask = get_pset_directional_mask(
            processed_pset, map_descriptor.spin_phase
        )
        project_pset_to_map(processed_pset, output_map, directional_mask, cg_correct)

    return output_map


def process_single_pset(
    pset: xr.Dataset,
    efficiency_data: pd.DataFrame,
    species: str,
    cg_correct: bool = False,
) -> xr.Dataset:
    """
    Process a single pointing set for projection to the sky map.

    Parameters
    ----------
    pset : xr.Dataset
        Single pointing set dataset to process.
    efficiency_data : pd.DataFrame
        Efficiency factor data for correcting counts.
    species : str
        The species to process (e.g., "h", "o").
    cg_correct : bool
        Whether to apply the CG correction to each PSET. A value of True will
        cause the pre-projection Compton Getting Correction to be applied to
        the PSET data.

    Returns
    -------
    xr.Dataset
        Processed pointing set ready for projection with efficiency corrections applied.
    """
    # Step 1: Normalize coordinate system
    pset_processed = normalize_pset_coordinates(pset, species)

    # Step 2: Add efficiency factors
    pset_processed = add_efficiency_factors_to_pset(pset_processed, efficiency_data)

    # Step 3: Calculate efficiency-corrected quantities
    pset_processed = calculate_efficiency_corrected_quantities(pset_processed)

    # Step 4: Add s/c velocity, optionally apply CG correction, and calculate
    # ram-mask
    pset_processed = add_spacecraft_velocity_to_pset(pset_processed)

    if cg_correct:
        # NOTE: Heliospheric frame energy selection for CG correction
        # The heliospheric (HF) energies passed to the CG correction algorithm
        # could in principle be completely different from the ESA central energies.
        # However, for Lo, the instrument team has chosen to use the same HF
        # energies as the ESA central energies (from the geometric factor files).
        # This decision aligns the energy grid between the spacecraft frame and
        # heliospheric frame representations.

        # Convert energy coordinate from keV to eV for CG correction
        # (energy coordinate was set in normalize_pset_coordinates in keV)
        energy_values_ev: xr.DataArray = pset_processed["energy"] * 1000.0
        pset_processed = apply_compton_getting_correction(
            pset_processed, energy_values_ev
        )
        # Prepare energy_sc for exposure time weighted projection
        pset_processed["energy_sc_exposure_factor"] = (
            pset_processed["energy_sc"] * pset_processed["exposure_factor"]
        )

    # Always calculate ram-mask to identify ram/anti-ram bins
    pset_processed = calculate_ram_mask(pset_processed)

    return pset_processed


def normalize_pset_coordinates(pset: xr.Dataset, species: str) -> xr.Dataset:
    """
    Normalize pointing set coordinates to match the output map.

    Parameters
    ----------
    pset : xr.Dataset
        Input pointing set dataset with potentially mismatched coordinates.
    species : str
        The species to process (e.g., "h", "o").

    Returns
    -------
    xr.Dataset
        Pointing set with normalized energy coordinates and dimension names.
    """
    # Load true energy values for this species (in keV, matching map convention)
    # TODO: Figure out how to handle esa_mode properly
    if "esa_mode" in pset:
        esa_mode = pset["esa_mode"].values[0]
    else:
        # Default to mode 0 if not available (HiRes mode)
        esa_mode = 0
    gf_dataset = reduce_geometric_factor_dataset(species, esa_mode=esa_mode)

    # Ensure consistent energy coordinates (maps want energy not esa_energy_step)
    pset_renamed = pset.rename_dims({"esa_energy_step": "energy"})

    # Drop the esa_energy_step coordinate first to avoid conflicts
    pset_renamed = pset_renamed.drop_vars("esa_energy_step")

    # Assign TRUE energy values as coordinates (in keV, matching map convention)
    pset_renamed = pset_renamed.assign_coords(energy=gf_dataset["Cntr_E"].values)

    # Rename the variables in the pset for projection to the map
    # L2 wants different variable names than l1c
    rename_map = {
        "exposure_time": "exposure_factor",
        f"{species}_counts": "counts",
        f"{species}_background_rates": "bg_rate",
        f"{species}_background_rates_stat_uncert": "bg_rate_stat_uncert",
    }
    pset_renamed = pset_renamed.rename_vars(rename_map)

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


def calculate_efficiency_corrected_quantities(
    pset: xr.Dataset,
) -> xr.Dataset:
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
    # counts / efficiency
    pset["counts_over_eff"] = pset["counts"] / pset["efficiency"]
    # counts / efficiency**2 (for variance propagation)
    pset["counts_over_eff_squared"] = pset["counts"] / (pset["efficiency"] ** 2)

    # background * exposure_factor for weighted average
    pset["bg_rate_exposure_factor"] = pset["bg_rate"] * pset["exposure_factor"]
    # background_uncertainty ** 2 * exposure_factor ** 2
    pset["bg_rate_stat_uncert_exposure_factor2"] = (
        pset["bg_rate_stat_uncert"] ** 2 * pset["exposure_factor"] ** 2
    )

    return pset


def project_pset_to_map(
    pset: xr.Dataset,
    output_map: AbstractSkyMap,
    directional_mask: xr.DataArray,
    cg_correct: bool = False,
) -> None:
    """
    Project pointing set data to the output map.

    Parameters
    ----------
    pset : xr.Dataset
        Processed pointing set ready for projection.
    output_map : AbstractSkyMap
        Target sky map to receive the projected data.
    directional_mask : xr.DataArray
        Boolean mask indicating which PSET bins to use for projection. This is
        how ram/anti-ram bins are removed depending on the descriptor spin phase.
    cg_correct : bool
        Whether the CG correction is being applied. If set to True, "energy_sc"
        is added to the list of variables to be projected.

    Returns
    -------
    None
        Function modifies output_map in place.
    """
    # Define base quantities to project
    value_keys = [
        "exposure_factor",
        "counts",
        "counts_over_eff",
        "counts_over_eff_squared",
        "bg_rate",
        "bg_rate_stat_uncert",
        "bg_rate_exposure_factor",
        "bg_rate_stat_uncert_exposure_factor2",
    ]
    if cg_correct:
        value_keys.append("energy_sc_exposure_factor")

    # Create LoPointingSet and project to map
    lo_pset = ena_maps.LoPointingSet(pset)
    output_map.project_pset_values_to_map(
        pointing_set=lo_pset,
        value_keys=value_keys,
        index_match_method=ena_maps.IndexMatchMethod.PUSH,
        pset_valid_mask=directional_mask,
    )
    logger.debug(f"Projected {len(value_keys)} quantities to sky map")


# =============================================================================
# GEOMETRIC FACTORS
# =============================================================================


def add_geometric_factors(dataset: xr.Dataset, species: str) -> xr.Dataset:
    """
    Add geometric factors to the sky map after projection.

    Parameters
    ----------
    dataset : xr.Dataset
        Sky map dataset to add geometric factors to.
    species : str
        The species to process (only "h" and "o" have geometric factors).

    Returns
    -------
    xr.Dataset
        Dataset with geometric factor variables added for the specified species.
    """
    # Only add geometric factors for hydrogen and oxygen
    if species not in ["h", "o"]:
        logger.warning(f"No geometric factors to add for species: {species}")
        return dataset

    logger.info(f"Loading and applying geometric factors for species: {species}")

    # Initialize geometric factor variables
    dataset = initialize_geometric_factor_variables(dataset)

    # Populate geometric factors for each energy step
    dataset = populate_geometric_factors(dataset, species)

    return dataset


def load_geometric_factor_data(species: str) -> pd.DataFrame:
    """
    Load geometric factor data for the specified species.

    Parameters
    ----------
    species : str
        The species to load geometric factors for ("h" or "o").

    Returns
    -------
    pd.DataFrame
        Geometric factor dataframe for the specified species.

    Raises
    ------
    ValueError
        If species is not "h" or "o".
    """
    if species not in ["h", "o"]:
        raise ValueError(
            f"Geometric factors only available for 'h' and 'o', got '{species}'"
        )

    anc_path = Path(__file__).parent.parent / "ancillary_data"

    if species == "h":
        gf_file = anc_path / "imap_lo_hydrogen-geometric-factor_v001.csv"
    else:  # species == "o"
        gf_file = anc_path / "imap_lo_oxygen-geometric-factor_v001.csv"

    return lo_ancillary.read_ancillary_file(gf_file)


def reduce_geometric_factor_dataset(species: str, esa_mode: int) -> xr.Dataset:
    """
    Get geometric factor data as xarray Dataset for a specific species and ESA mode.

    This helper function loads geometric factor data, filters by ESA mode, converts
    to xarray, and selects all 7 energy steps for vectorized operations.

    Parameters
    ----------
    species : str
        The species to load geometric factors for ("h" or "o").
    esa_mode : int
        ESA mode (0 for HiRes, 1 for HiThr).

    Returns
    -------
    xarray.Dataset
        Geometric factor data indexed by Observed_E-Step (1-7), containing all
        columns from the geometric factor CSV file.
    """
    # Load geometric factor data for this species
    gf_data = load_geometric_factor_data(species)

    # Filter for the specific ESA mode
    if "esa_mode" in gf_data.columns:
        gf_data = gf_data[gf_data["esa_mode"] == esa_mode].copy()

    # Convert to xarray Dataset indexed by energy step for vectorized selection
    gf_ds = gf_data.set_index("Observed_E-Step").to_xarray()

    # Lo Instrument team: Use only geometric factors where
    # incident_E-Step == Observed_E-Step
    gf_ds = gf_ds.where(gf_ds["incident_E-Step"] == gf_ds["Observed_E-Step"], drop=True)

    # Select energy steps 1-7 and return
    return gf_ds.sel({"Observed_E-Step": range(1, 8)})


def initialize_geometric_factor_variables(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Initialize geometric factor variables for the specified species.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset to add geometric factor variables to.

    Returns
    -------
    xr.Dataset
        Dataset with initialized geometric factor variables for the specified species.
    """
    gf_vars = [
        "energy",
        "energy_delta_minus",
        "energy_delta_plus",
        "geometric_factor",
        "geometric_factor_stat_uncert",
    ]

    # Initialize variables with proper dimensions (energy only)
    for var in gf_vars:
        dataset[var] = xr.DataArray(
            np.zeros(7),
            dims=["energy"],
        )

    return dataset


def populate_geometric_factors(
    dataset: xr.Dataset,
    species: str,
) -> xr.Dataset:
    """
    Populate geometric factor values for each energy step.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with initialized geometric factor variables.
    species : str
        The species to process (only "h" and "o" have geometric factors).

    Returns
    -------
    xr.Dataset
        Dataset with populated geometric factor values for the specified species.
    """
    # Only populate if the species has geometric factors
    if species not in ["h", "o"]:
        logger.debug(f"No geometric factors to populate for species: {species}")
        return dataset

    # Mapping of dataset variables to dataframe columns for this species
    gf_coords = {"energy": "Cntr_E"}
    gf_vars = {
        "geometric_factor": f"GF_Trpl_{species.upper()}",
        "geometric_factor_stat_uncert": f"GF_Trpl_{species.upper()}_unc",
    }
    if species == "h":
        # NOTE: From an e-mail from Nathan on 2025-09-11 (values converted to keV)
        energy_delta_hires_values = (
            np.array([5.43, 10.02, 18.61, 33.31, 64.98, 131.64, 262.35]) * 1e-3
        )
        energy_delta_hithr_values = (
            np.array([8.81, 16.04, 28.50, 53.13, 105.60, 219.67, 413.60]) * 1e-3
        )
    else:  # species == "o"
        energy_delta_hires_values = (
            np.array([5.82, 11.10, 21.78, 41.47, 85.61, 180.67, 361.93]) * 1e-3
        )
        energy_delta_hithr_values = (
            np.array([9.45, 17.84, 33.51, 66.61, 139.95, 302.24, 569.48]) * 1e-3
        )

    # Get ESA mode from the map (assuming it's constant or we take the first)
    # TODO: Figure out how to handle esa_mode properly
    if "esa_mode" in dataset:
        esa_mode = dataset["esa_mode"].values[0]
    else:
        # Default to mode 0 if not available (HiRes mode)
        esa_mode = 0

    # Filter for the specific ESA mode
    gf_dataset = reduce_geometric_factor_dataset(species, esa_mode)

    # Populate geometric factors in dataset
    dataset = dataset.assign_coords(energy=gf_dataset[gf_coords["energy"]].values)
    for var, col in gf_vars.items():
        dataset[var].values = gf_dataset[col].values

    # Update delta_minus and delta_plus based on ESA mode
    # converting eV to keV
    if esa_mode == 0:  # HiRes
        dataset["energy_delta_minus"].values = energy_delta_hires_values
        dataset["energy_delta_plus"].values = energy_delta_hires_values
    else:  # HiThr
        dataset["energy_delta_minus"].values = energy_delta_hithr_values
        dataset["energy_delta_plus"].values = energy_delta_hithr_values

    return dataset


# =============================================================================
# RATES AND INTENSITIES CALCULATIONS
# =============================================================================


def calculate_all_rates_and_intensities(
    dataset: xr.Dataset,
    sputtering_correction: bool = False,
    bootstrap_correction: bool = False,
    flux_correction: bool = False,
    o_map_dataset: xr.Dataset | None = None,
    flux_factors: Path | None = None,
    cg_correction: bool = False,
) -> xr.Dataset:
    """
    Calculate rates and intensities with proper error propagation.

    Parameters
    ----------
    dataset : xr.Dataset
        Sky map dataset with count data and geometric factors.
    sputtering_correction : bool, optional
        Whether to apply sputtering corrections to oxygen intensities.
        Default is False.
    bootstrap_correction : bool, optional
        Whether to apply bootstrap corrections to intensities.
        Default is False.
    flux_correction : bool, optional
        Whether to apply flux corrections to intensities.
        Default is False.
    o_map_dataset : xr.Dataset, optional
        Dataset specifically for oxygen, needed for sputtering corrections.
    flux_factors : Path, optional
        Path to flux factor file for flux corrections.
    cg_correction : bool, optional
        Whether to apply CG correction to intensities.

    Returns
    -------
    xr.Dataset
        Dataset with calculated rates, intensities, and uncertainties for the
        specified species.
    """
    # Step 1: Calculate rates for the specified species
    dataset = calculate_rates(dataset)

    # Step 2: Calculate intensities
    dataset = calculate_intensities(dataset)

    # Step 3: Calculate background rates and intensities
    dataset = calculate_backgrounds(dataset)

    # Optional Step 4: Calculate sputtering corrections
    if sputtering_correction:
        logger.info("Calculating sputtering corrections")
        dataset = calculate_sputtering_corrections(dataset, o_map_dataset)

    # Optional Step 5: Calculate bootstrap corrections
    if bootstrap_correction:
        logger.info("Calculating bootstrap corrections")
        dataset = calculate_bootstrap_corrections(dataset)

    # Optional Step 6: Calculate flux corrections
    if flux_correction:
        if flux_factors is None:
            raise ValueError("Flux factors file must be provided for flux corrections")
        dataset = calculate_flux_corrections(dataset, flux_factors)

    # Optional Step 7: Finish CG correction
    if cg_correction:
        logger.info("Interpolating map intensities to helio-frame energies")
        # Finish calculation of the exposure factor weighted projection of energy_sc
        # and convert to units of keV
        dataset["energy_sc"] = (
            dataset["energy_sc_exposure_factor"] / dataset["exposure_factor"] / 1e3
        )
        dataset = interpolate_map_flux_to_helio_frame(
            dataset,
            dataset["energy"],
            dataset["energy"],
            ["ena_intensity", "bg_intensity"],
        )

    # Step 7: Clean up intermediate variables
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
        for the specified species.
    """
    # Rate = counts / exposure_factor
    # TODO: Account for ena / isn naming differences
    dataset["ena_count_rate"] = dataset["counts"] / dataset["exposure_factor"]

    # Poisson uncertainty on the counts propagated to the rate
    # TODO: Is there uncertainty in the exposure time too?
    dataset["ena_count_rate_stat_uncert"] = (
        np.sqrt(dataset["counts"]) / dataset["exposure_factor"]
    )

    return dataset


def calculate_intensities(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate particle intensities and uncertainties for the specified species.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with count rates, geometric factors, and center energies.

    Returns
    -------
    xr.Dataset
        Dataset with calculated particle intensities and their statistical
        and systematic uncertainties for the specified species.
    """
    # Equation 3 from mapping document (average intensity)
    dataset["ena_intensity"] = dataset["counts_over_eff"] / (
        dataset["geometric_factor"] * dataset["energy"] * dataset["exposure_factor"]
    )

    # Equation 4 from mapping document (statistical uncertainty)
    # Note that we need to take the square root to get the uncertainty as
    # the equation is for the variance
    dataset["ena_intensity_stat_uncert"] = np.sqrt(
        dataset["counts_over_eff_squared"]
    ) / (dataset["geometric_factor"] * dataset["energy"] * dataset["exposure_factor"])

    # Equation 5 from mapping document (systematic uncertainty)
    dataset["ena_intensity_sys_err"] = (
        dataset["ena_intensity"]
        * dataset["geometric_factor_stat_uncert"]
        / dataset["geometric_factor"]
    )

    return dataset


def calculate_backgrounds(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate background rates and intensities for the specified species.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with count rates, geometric factors, and center energies.

    Returns
    -------
    xr.Dataset
        Dataset with calculated background rates and intensities for the
        specified species.
    """
    # Equation 6 from mapping document (background rate)
    # exposure time weighted average of the background rates
    dataset["bg_rate"] = dataset["bg_rate_exposure_factor"] / dataset["exposure_factor"]
    # Equation 7 from mapping document (background statistical uncertainty)
    dataset["bg_rate_stat_uncert"] = np.sqrt(
        dataset["bg_rate_stat_uncert_exposure_factor2"]
        / dataset["exposure_factor"] ** 2
    )
    # Equation 8 from mapping document (background systematic uncertainty)
    dataset["bg_rate_sys_err"] = (
        dataset["bg_rate"]
        * dataset["geometric_factor_stat_uncert"]
        / dataset["geometric_factor"]
    )

    # Background intensity
    dataset["bg_intensity"] = dataset["bg_rate"] / (
        dataset["geometric_factor"] * dataset["energy"]
    )
    dataset["bg_intensity_stat_uncert"] = dataset["bg_rate_stat_uncert"] / (
        dataset["geometric_factor"] * dataset["energy"]
    )
    dataset["bg_intensity_sys_err"] = (
        dataset["bg_intensity"]
        * dataset["geometric_factor_stat_uncert"]
        / dataset["geometric_factor"]
    )

    return dataset


def calculate_sputtering_corrections(
    dataset: xr.Dataset, o_dataset: xr.Dataset
) -> xr.Dataset:
    """
    Calculate sputtering corrections from oxygen intensities.

    Only for Oxygen sputtering and correction only at ESA levels 5 and 6
    for 90 degree maps. If off-angle maps are made, we may have to extend
    this to levels 3 and 4 as well.

    Follows equations 9-13 from the mapping document.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with count rates, geometric factors, and center energies.
        This is an H dataset that we are applying the corrections to.
    o_dataset : xr.Dataset
        Dataset specifically for oxygen, needed to access oxygen intensities
        and uncertainties.

    Returns
    -------
    xr.Dataset
        Dataset with calculated sputtering-corrected intensities and their
        uncertainties.
    """
    logger.info("Applying sputtering corrections to hydrogen intensities")
    # Only apply sputtering correction to esa levels 5 and 6 (indices 4 and 5)
    energy_indices = [4, 5]
    small_dataset = dataset.isel(epoch=0, energy=energy_indices)
    o_small_dataset = o_dataset.isel(epoch=0, energy=energy_indices)

    # We need to align the energy dimensions from the oxygen dataset to the
    # Hydrogen dataset so the calculations below get aligned by xarray correctly.
    o_small_dataset["energy"] = small_dataset["energy"]

    # Equation 9
    j_o_prime = o_small_dataset["ena_intensity"] - o_small_dataset["bg_intensity"]
    j_o_prime.values[j_o_prime.values < 0] = 0  # No negative intensities
    j_o_prime_valid = np.isfinite(j_o_prime) & (j_o_prime > 0)

    # Equation 10
    j_o_prime_var = (
        o_small_dataset["ena_intensity_stat_uncert"] ** 2
        + o_small_dataset["bg_intensity_stat_uncert"] ** 2
    )

    # NOTE: From table 2 of the mapping document, for energy level 5 and 6
    sputter_correction_factor = xr.DataArray(
        [0.15, 0.01], dims=["energy"], coords={"energy": small_dataset["energy"]}
    )
    # Equation 11
    # Remove the sputtered oxygen intensity to correct the original H intensity
    sputter_corrected_intensity = xr.where(
        j_o_prime_valid,
        small_dataset["ena_intensity"] - sputter_correction_factor * j_o_prime,
        small_dataset["ena_intensity"],
    )

    # Equation 12
    sputter_corrected_intensity_var = xr.where(
        j_o_prime_valid,
        small_dataset["ena_intensity_stat_uncert"] ** 2
        + (sputter_correction_factor**2) * j_o_prime_var,
        small_dataset["ena_intensity_stat_uncert"] ** 2,
    )

    # Equation 13
    sputter_corrected_intensity_sys_err = xr.where(
        j_o_prime_valid,
        sputter_corrected_intensity
        / small_dataset["ena_intensity"]
        * small_dataset["ena_intensity_sys_err"],
        small_dataset["ena_intensity_sys_err"],
    )

    # Now put the corrected values into the original dataset
    dataset["ena_intensity"].values[0, energy_indices, ...] = (
        sputter_corrected_intensity.values
    )
    dataset["ena_intensity_stat_uncert"].values[0, energy_indices, ...] = np.sqrt(
        sputter_corrected_intensity_var.values
    )
    dataset["ena_intensity_sys_err"].values[0, energy_indices, ...] = (
        sputter_corrected_intensity_sys_err.values
    )

    return dataset


def calculate_bootstrap_corrections(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate bootstrap corrections for hydrogen and oxygen intensities.

    Follows equations 14-35 from the mapping document.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with count rates, geometric factors, and center energies.

    Returns
    -------
    xr.Dataset
        Dataset with calculated bootstrap-corrected intensities and their
        uncertainties for hydrogen.
    """
    logger.info("Applying bootstrap corrections")

    # Table 3 bootstrap terms h_i,k - convert to xarray for better dimension handling
    bootstrap_factor_array = np.array(
        [
            [0, 0.03, 0.01, 0, 0, 0, 0, 0],
            [0, 0, 0.05, 0.02, 0.01, 0, 0, 0],
            [0, 0, 0, 0.09, 0.03, 0.016, 0.01, 0],
            [0, 0, 0, 0, 0.16, 0.068, 0.016, 0.01],
            [0, 0, 0, 0, 0, 0.29, 0.068, 0.016],
            [0, 0, 0, 0, 0, 0, 0.52, 0.061],
            [0, 0, 0, 0, 0, 0, 0, 0.75],
        ]
    )
    # Create xarray DataArray with named dimensions for proper broadcasting
    bootstrap_factor = xr.DataArray(
        bootstrap_factor_array,
        dims=["energy_i", "energy_k"],
        coords={
            "energy_i": dataset["energy"].values,
            # Add an extra coordinate for the virtual E8 channel, unused
            # in the broadcasting calculations
            "energy_k": np.concatenate([dataset["energy"].values, [np.nan]]),
        },
    )

    # Equation 14
    j_c_prime = dataset["ena_intensity"] - dataset["bg_intensity"]
    j_c_prime.values[j_c_prime.values < 0] = 0

    # Equation 15
    j_c_prime_var = dataset["ena_intensity_stat_uncert"] ** 2

    # Equation 16 - systematic error propagation
    # Handle division by zero: only compute where ena_intensity > 0
    j_c_prime_err = xr.where(
        dataset["ena_intensity"] > 0,
        j_c_prime / dataset["ena_intensity"] * dataset["ena_intensity_sys_err"],
        0,
    )

    # NOTE: E8 virtual channel calculation is from the text. This is to
    # start the calculations off from the higher energies and avoid
    # reliance on IMAP Hi energy channels.
    # E8 is a virtual energy channel at 2.1 * E7
    e8 = 2.1 * dataset["energy"].values[-1]

    j_c_6 = j_c_prime.isel(energy=5)
    j_c_7 = j_c_prime.isel(energy=6)
    e_6 = dataset["energy"].isel(energy=5)
    e_7 = dataset["energy"].isel(energy=6)

    # Calculate gamma, ignoring any invalid values
    # Fill in the invalid values with zeros after the fact
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.log(j_c_6 / j_c_7) / np.log(e_6 / e_7)
        j_8_b = j_c_7 * (e8 / e_7) ** gamma

    # Set j_8_b to zero where the calculation was invalid
    j_8_b = j_8_b.where(np.isfinite(j_8_b) & (j_8_b > 0), 0)

    # Initialize bootstrap intensity and uncertainty arrays
    dataset["bootstrap_intensity"] = xr.zeros_like(dataset["ena_intensity"])
    dataset["bootstrap_intensity_var"] = xr.zeros_like(dataset["ena_intensity"])
    dataset["bootstrap_intensity_sys_err"] = xr.zeros_like(dataset["ena_intensity"])

    for i in range(6, -1, -1):
        # Create views for the current energy channel to avoid repeated indexing
        bootstrap_intensity_i = dataset["bootstrap_intensity"][0, i, ...]
        bootstrap_intensity_var_i = dataset["bootstrap_intensity_var"][0, i, ...]
        j_c_prime_i = j_c_prime[0, i, ...]
        j_c_prime_var_i = j_c_prime_var[0, i, ...]

        # Initialize the variable with the non-summation term and virtual
        # channel energy subtraction first, then iterate through the other
        # channels which can be looked up via indexing
        # i.e. the summation is always k=i+1 to 7, because we've already
        # included the k=8 term here.
        # NOTE: The paper uses 1-based indexing and we use 0-based indexing
        #       so there is an off-by-one difference in the indices.
        bootstrap_intensity_i[:] = (
            j_c_prime_i - bootstrap_factor.isel(energy_i=i, energy_k=7) * j_8_b[0, ...]
        )
        # NOTE: We will square root at the end to get the uncertainty, but
        #       all equations are with variances
        bootstrap_intensity_var_i[:] = j_c_prime_var_i

        # Vectorized summation using xarray's built-in broadcasting
        # Select the relevant k indices for summation (k = i+1 to 6)
        k_indices = list(range(i + 1, 7))

        # Get bootstrap factors for this i and the relevant k values
        # Rename energy_k dimension to energy for alignment with intensity
        bootstrap_factors_k = bootstrap_factor.isel(
            energy_i=i, energy_k=k_indices
        ).rename({"energy_k": "energy"})

        # Get intensity slices - these will have an 'energy' dimension still
        intensity_k = dataset["bootstrap_intensity"][0, k_indices, ...]
        intensity_var_k = dataset["bootstrap_intensity_var"][0, k_indices, ...]

        # Subtraction terms from equations 18-23 (xarray vectorized)
        bootstrap_intensity_i -= (bootstrap_factors_k * intensity_k).sum(dim="energy")

        # Summation terms from equations 25-30 (xarray vectorized)
        bootstrap_intensity_var_i += (bootstrap_factors_k**2 * intensity_var_k).sum(
            dim="energy"
        )

        # Again zero any bootstrap fluxes that are negative
        bootstrap_intensity_i.values[bootstrap_intensity_i < 0] = 0.0

    # Equation 31 - systematic error propagation for bootstrap intensity
    # Handle division by zero: only compute where j_c_prime > 0
    dataset["bootstrap_intensity_sys_err"] = xr.where(
        j_c_prime > 0, dataset["bootstrap_intensity"] / j_c_prime * j_c_prime_err, 0
    )

    valid_bootstrap = (dataset["bootstrap_intensity"] > 0) & np.isfinite(
        dataset["bootstrap_intensity"]
    )
    # Update the original intensity values
    # Equation 32 / 33
    # ena_intensity = ena_intensity (J_c) - (j_c_prime - J_b)
    dataset["ena_intensity"] = xr.where(
        valid_bootstrap,
        dataset["ena_intensity"] - j_c_prime + dataset["bootstrap_intensity"],
        dataset["ena_intensity"],
    )

    # Ensure corrected intensities are non-negative
    dataset["ena_intensity"] = xr.where(
        dataset["ena_intensity"] < 0, 0, dataset["ena_intensity"]
    )

    # Equation 34 - statistical uncertainty
    # Take the square root, since we were in variances up to this point
    dataset["ena_intensity_stat_uncert"] = xr.where(
        valid_bootstrap,
        np.sqrt(dataset["bootstrap_intensity_var"]),
        dataset["ena_intensity_stat_uncert"],
    )

    # Equation 35 - systematic error for corrected intensity
    # Handle division by zero and ensure reasonable values
    dataset["ena_intensity_sys_err"] = xr.zeros_like(dataset["ena_intensity"])

    # Only compute where bootstrap intensity is valid
    dataset["ena_intensity_sys_err"] = xr.where(
        valid_bootstrap,
        (
            dataset["ena_intensity"]
            / dataset["bootstrap_intensity"]
            * dataset["bootstrap_intensity_sys_err"]
        ),
        0,
    )

    # Drop the intermediate bootstrap variables
    dataset = dataset.drop_vars(
        [
            "bootstrap_intensity",
            "bootstrap_intensity_var",
            "bootstrap_intensity_sys_err",
        ]
    )

    return dataset


def calculate_flux_corrections(dataset: xr.Dataset, flux_factors: Path) -> xr.Dataset:
    """
    Calculate flux corrections for intensities.

    Uses the shared ena maps ``PowerLawFluxCorrector`` class to do the
    correction calculations.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with count rates, geometric factors, and center energies.
    flux_factors : Path
        Path to the eta flux factor file to use for corrections. Read in as
        an ancillary file in the preprocessing step.

    Returns
    -------
    xr.Dataset
        Dataset with calculated flux-corrected intensities and their
        uncertainties for the specified species.
    """
    logger.info("Applying flux corrections")

    # Flux correction
    corrector = PowerLawFluxCorrector(flux_factors)

    # NOTE: We need to apply this to both total flux and background flux
    for var in ["ena", "bg"]:
        # Apply flux correction with xarray inputs
        dataset[f"{var}_intensity"], dataset[f"{var}_intensity_stat_uncert"] = (
            corrector.apply_flux_correction(
                dataset[f"{var}_intensity"],
                dataset[f"{var}_intensity_stat_uncert"],
                dataset["energy"],
            )
        )

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

    # Only remove variables that exist in the dataset for the specific species
    potential_vars = [
        "geometric_factor",
        "geometric_factor_stat_uncert",
        "counts_over_eff",
        "counts_over_eff_squared",
        "bg_rate_exposure_factor",
        "bg_rate_stat_uncert_exposure_factor2",
    ]

    for potential_var in potential_vars:
        if potential_var in dataset.data_vars:
            vars_to_remove.append(potential_var)

    return dataset.drop_vars(vars_to_remove)
