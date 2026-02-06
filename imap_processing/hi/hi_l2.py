"""IMAP-HI L2 processing module."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps.ena_maps import (
    HiPointingSet,
    RectangularSkyMap,
)
from imap_processing.ena_maps.utils.corrections import (
    PowerLawFluxCorrector,
    add_spacecraft_velocity_to_pset,
    apply_compton_getting_correction,
    calculate_ram_mask,
    get_pset_directional_mask,
    interpolate_map_flux_to_helio_frame,
)
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.hi.utils import CalibrationProductConfig

logger = logging.getLogger(__name__)

SC_FRAME_VARS_TO_PROJECT = {
    "counts",
    "exposure_factor",
    "bg_rates",
    "bg_rates_unc",
    "obs_date",
}
HELIO_FRAME_VARS_TO_PROJECT = SC_FRAME_VARS_TO_PROJECT | {"energy_sc"}
# TODO: is an exposure time weighted average for obs_date appropriate?
FULL_EXPOSURE_TIME_AVERAGE_SET = {"bg_rates", "bg_rates_unc", "obs_date", "energy_sc"}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def hi_l2(
    psets: list[str | Path],
    l2_ancillary_path_dict: dict[str, Path],
    descriptor: str,
) -> list[xr.Dataset]:
    """
    Process IMAP-Hi L1C data into L2 CDF data products.

    This is the main entry point for L2 processing. It orchestrates the entire
    processing pipeline from L1C pointing sets to L2 sky maps with intensities.

    Parameters
    ----------
    psets : list of str or pathlib.Path
        List of input PSETs to make a map from.
    l2_ancillary_path_dict : dict[str, pathlib.Path]
        Mapping containing ancillary file descriptors as keys and file paths as
        values. Required keys are: ["cal-prod", "esa-energies", "esa-eta-fit-factors"].
    descriptor : str
        The map descriptor to be produced
        (e.g., "h90-ena-h-sf-nsp-full-hae-6deg-3mo").

    Returns
    -------
    list[xarray.Dataset]
        List containing the processed L2 dataset with rates, intensities,
        and uncertainties.

    Raises
    ------
    ValueError
        If map_descriptor sensor attribute is invalid.
    NotImplementedError
        If HEALPix map output is requested (only rectangular maps supported).
    """
    logger.info("Starting IMAP-Hi L2 processing pipeline")
    logger.info(f"Descriptor: {descriptor}")
    logger.info(f"Processing {len(psets)} pointing sets")

    # Parse the map descriptor
    map_descriptor = MapDescriptor.from_string(descriptor)
    if not isinstance(map_descriptor.sensor, str):
        raise ValueError(
            "Invalid map_descriptor. Sensor attribute must be of type str "
            "and be either '45' or '90'"
        )

    logger.info(f"Step 1: Creating sky map from {len(psets)} pointing sets")
    sky_map = create_sky_map_from_psets(
        psets,
        l2_ancillary_path_dict,
        map_descriptor,
    )

    logger.info("Step 2: Calculating rates and intensities")
    sky_map.data_1d = calculate_all_rates_and_intensities(
        sky_map.data_1d,
        l2_ancillary_path_dict,
        map_descriptor,
    )

    logger.info("Step 3: Finalizing dataset with attributes")
    l2_ds = sky_map.build_cdf_dataset(
        "hi",
        "l2",
        descriptor,
        sensor=map_descriptor.sensor,
    )

    logger.info("IMAP-Hi L2 processing pipeline completed successfully")
    return [l2_ds]


# =============================================================================
# SKY MAP CREATION PIPELINE
# =============================================================================


def create_sky_map_from_psets(
    psets: list[str | Path],
    l2_ancillary_path_dict: dict[str, Path],
    descriptor: MapDescriptor,
) -> RectangularSkyMap:
    """
    Project Hi PSET data into a sky map.

    Parameters
    ----------
    psets : list of str or pathlib.Path
        List of input PSETs to make a map from.
    l2_ancillary_path_dict : dict[str, pathlib.Path]
        Mapping containing ancillary file descriptors as keys and file paths as
        values. Require keys are: ["cal-prod", "esa-energies", "esa-eta-fit-factors"].
    descriptor : imap_processing.ena_maps.utils.naming.MapDescriptor
        Output filename descriptor. Contains full configuration for the options
        of how to generate the map.

    Returns
    -------
    sky_map : RectangularSkyMap
        The sky map with all the PSET data projected into the map. Includes
        an energy coordinate and energy_delta_minus and energy_delta_plus
        variables from ESA energy calibration data.
    """
    if len(psets) == 0:
        raise ValueError("No PSETs provided for map creation")

    output_map = descriptor.to_empty_map()

    if not isinstance(output_map, RectangularSkyMap):
        raise NotImplementedError("Healpix map output not supported for Hi")

    vars_to_bin = (
        HELIO_FRAME_VARS_TO_PROJECT
        if descriptor.frame_descriptor == "hf"
        else SC_FRAME_VARS_TO_PROJECT
    )
    vars_to_exposure_time_average = FULL_EXPOSURE_TIME_AVERAGE_SET & vars_to_bin

    for i_pset, pset_path in enumerate(psets):
        logger.debug(f"Processing {pset_path}")
        pset_ds = load_cdf(pset_path)

        # Store the first PSET esa_energy_step values and make sure every PSET
        # contains the same set of esa_energy_step values.
        # TODO: Correctly handle PSETs with different esa_energy_step values.
        if i_pset == 0:
            cached_esa_steps = pset_ds["esa_energy_step"].values.copy()
            esa_ds = esa_energy_df(
                l2_ancillary_path_dict["esa-energies"],
                pset_ds["esa_energy_step"].values,
            ).to_xarray()
            energy_kev = esa_ds["nominal_central_energy"]
        if not np.array_equal(cached_esa_steps, pset_ds["esa_energy_step"].values):
            raise ValueError(
                "All PSETs must have the same set of esa_energy_step values."
            )

        pset_processed = process_single_pset(
            pset_ds,
            energy_kev,
            descriptor,
            vars_to_exposure_time_average,
        )

        # Project (bin) the PSET variables into the map pixels
        directional_mask = get_pset_directional_mask(
            pset_processed, descriptor.spin_phase
        )
        hi_pset = HiPointingSet(pset_processed)
        output_map.project_pset_values_to_map(
            hi_pset, list(vars_to_bin), pset_valid_mask=directional_mask
        )

    # Finish the exposure time weighted mean calculation of backgrounds
    # Allow divide by zero to fill set pixels with zero exposure time to NaN
    with np.errstate(divide="ignore"):
        for var in vars_to_exposure_time_average:
            output_map.data_1d[var] /= output_map.data_1d["exposure_factor"]

    # Add ESA energy data to the map dataset for use in rate/intensity calculations
    energy_delta = esa_ds["bandpass_fwhm"] / 2
    output_map.data_1d["energy_delta_minus"] = energy_delta
    output_map.data_1d["energy_delta_plus"] = energy_delta
    # Add energy as an auxiliary coordinate (keV values indexed by esa_energy_step)
    output_map.data_1d = output_map.data_1d.assign_coords(
        energy=("esa_energy_step", esa_ds["nominal_central_energy"].values)
    )

    return output_map


# =============================================================================
# PSET PROCESSING
# =============================================================================


def process_single_pset(
    pset: xr.Dataset,
    energy_kev: xr.DataArray,
    descriptor: MapDescriptor,
    vars_to_exposure_time_average: set[str],
) -> xr.Dataset:
    """
    Process a single pointing set for projection to the sky map.

    Parameters
    ----------
    pset : xarray.Dataset
        Single pointing set dataset to process.
    energy_kev : xarray.DataArray
        Central energy values in keV for the ESA energy steps.
    descriptor : imap_processing.ena_maps.utils.naming.MapDescriptor
        Map descriptor containing processing configuration.
    vars_to_exposure_time_average : set of str
        Set of variable names that need to be multiplied by exposure factor
        for weighted averaging.

    Returns
    -------
    xarray.Dataset
        Processed pointing set ready for projection.
    """
    # Step 1: Rename some PSET vars to match L2 variables
    pset_processed = pset.rename(HiPointingSet.l1c_to_l2_var_mapping)

    # Step 2: Add obs_date variable to be used in determining a map mean obs_date
    mid_time = (
        pset_processed["epoch"].values[0] + pset_processed["epoch_delta"].values[0] / 2
    )
    pset_processed["obs_date"] = xr.full_like(
        pset_processed["exposure_factor"], float(mid_time)
    )

    # Step 3: Add spacecraft velocity
    pset_processed = add_spacecraft_velocity_to_pset(pset_processed)

    # Step 4: Optionally apply Compton-Getting correction for heliocentric frame
    if descriptor.frame_descriptor == "hf":
        # convert esa nominal central energy from keV to eV
        esa_energy_ev = energy_kev * 1000
        pset_processed = apply_compton_getting_correction(pset_processed, esa_energy_ev)

    # Step 5: Calculate ram mask
    pset_processed = calculate_ram_mask(pset_processed)

    # Step 6: Multiply variables that need to be exposure time weighted average by
    # exposure factor.
    for var in vars_to_exposure_time_average:
        if var in pset_processed:
            pset_processed[var] *= pset_processed["exposure_factor"]

    return pset_processed


# =============================================================================
# RATES AND INTENSITIES CALCULATIONS
# =============================================================================


def calculate_all_rates_and_intensities(
    map_ds: xr.Dataset,
    l2_ancillary_path_dict: dict[str, Path],
    descriptor: MapDescriptor,
) -> xr.Dataset:
    """
    Calculate rates and intensities with proper error propagation.

    This function orchestrates the full rate and intensity calculation pipeline
    including signal rates, intensities, coordinate transformations, and optional
    Compton-Getting corrections for heliocentric frame maps.

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset with projected PSET data (counts, exposure_factor, bg_rates,
        energy_delta_minus, energy_delta_plus, etc.) and an `energy` coordinate
        containing the ESA nominal central energies in keV.
    l2_ancillary_path_dict : dict[str, pathlib.Path]
        Mapping containing ancillary file descriptors as keys and file paths as
        values. Required keys are: ["cal-prod", "esa-energies", "esa-eta-fit-factors"].
    descriptor : imap_processing.ena_maps.utils.naming.MapDescriptor
        Map descriptor containing processing configuration.

    Returns
    -------
    map_ds : xarray.Dataset
        Map dataset with calculated rates, intensities, and uncertainties.
    """
    # Step 1: Calculate ENA signal rates
    logger.debug("Calculating ENA signal rates")
    map_ds = calculate_ena_signal_rates(map_ds)

    # Step 2: Calculate ENA intensities
    logger.debug("Calculating ENA intensities")
    map_ds = calculate_ena_intensity(map_ds, l2_ancillary_path_dict, descriptor)

    # Step 3: Handle obs_date variable type conversion
    # TODO: Handle variable types correctly in RectangularSkyMap.build_cdf_dataset
    obs_date = map_ds["obs_date"]
    # Replace non-finite values with the int64 sentinel before casting
    obs_date_filled = xr.where(
        np.isfinite(obs_date),
        obs_date,
        np.int64(-9223372036854775808),
    )
    map_ds["obs_date"] = obs_date_filled.astype("int64")
    # TODO: Figure out how to compute obs_date_range (stddev of obs_date)
    map_ds["obs_date_range"] = xr.zeros_like(map_ds["obs_date"])

    # Step 4: Swap esa_energy_step dimension for energy coordinate
    map_ds = map_ds.swap_dims({"esa_energy_step": "energy"})
    map_ds = map_ds.drop_vars(
        ["esa_energy_step", "esa_energy_step_label"], errors="ignore"
    )

    # Step 5: Apply Compton-Getting interpolation for heliocentric frame maps
    if descriptor.frame_descriptor == "hf":
        logger.debug("Applying Compton-Getting interpolation for heliocentric frame")
        # Convert energy coordinate from keV to eV for interpolation
        esa_energy_ev = map_ds["energy"] * 1000
        map_ds = interpolate_map_flux_to_helio_frame(
            map_ds,
            esa_energy_ev,  # ESA energies in eV
            esa_energy_ev,  # heliocentric energies (same as ESA energies)
            ["ena_intensity"],
        )
        # Drop any esa_energy_step_label that may have been re-added
        map_ds = map_ds.drop_vars(["esa_energy_step_label"], errors="ignore")

    return map_ds


def calculate_ena_signal_rates(map_ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the ENA signal rates.

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset that has counts, exposure_factor, and bg_rates calculated.

    Returns
    -------
    map_ds : xarray.Dataset
        Map dataset with new variables: ena_signal_rates, ena_signal_rate_stat_unc.
    """
    # Allow divide by zero to set pixels with zero exposure time to NaN
    with np.errstate(divide="ignore"):
        # Calculate the ENA Signal Rate
        map_ds["ena_signal_rates"] = (
            map_ds["counts"] / map_ds["exposure_factor"] - map_ds["bg_rates"]
        )
        # Calculate the ENA Signal Rate Uncertainties
        # The minimum count uncertainty is 1 for any pixel that has non-zero
        # exposure time. See IMAP Hi Algorithm Document section 3.1.1. Here,
        # we can ignore the non-zero exposure time condition when setting the
        # minimum count uncertainty because division by zero exposure time results
        # in the correct NaN value.
        min_counts_unc = xr.ufuncs.maximum(map_ds["counts"], 1)
        map_ds["ena_signal_rate_stat_unc"] = (
            np.sqrt(min_counts_unc) / map_ds["exposure_factor"]
        )

    # Statistical fluctuations may result in a negative ENA signal rate after
    # background subtraction. A negative signal rate is nonphysical. See IMAP Hi
    # Algorithm Document section 3.1.1
    map_ds["ena_signal_rates"].values[map_ds["ena_signal_rates"].values < 0] = 0

    return map_ds


def calculate_ena_intensity(
    map_ds: xr.Dataset,
    l2_ancillary_path_dict: dict[str, Path],
    descriptor: MapDescriptor,
) -> xr.Dataset:
    """
    Calculate the ena intensities.

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset that has ena_signal_rate fields calculated.
    l2_ancillary_path_dict : dict[str, pathlib.Path]
        Mapping containing ancillary file descriptors as keys and file paths as
        values. Require keys are: ["cal-prod", "esa-energies", "esa-eta-fit-factors"].
    descriptor : imap_processing.ena_maps.utils.naming.MapDescriptor
        Output filename descriptor. Contains full configuration for the options
        of how to generate the map. For this function, the principal data string
        is used to determine if a flux correction should be applied.

    Returns
    -------
    map_ds : xarray.Dataset
        Map dataset with new variables: ena_intensity, ena_intensity_stat_uncert,
        ena_intensity_sys_err.
    """
    # read calibration product configuration file
    cal_prod_df = CalibrationProductConfig.from_csv(l2_ancillary_path_dict["cal-prod"])
    # reindex_like removes esa_energy_steps and calibration products not in the
    # map_ds esa_energy_step and calibration_product coordinates
    geometric_factor = cal_prod_df.to_xarray().reindex_like(map_ds)["geometric_factor"]
    geometric_factor = geometric_factor.transpose(
        *[coord for coord in map_ds.coords if coord in geometric_factor.coords]
    )
    energy_df = esa_energy_df(
        l2_ancillary_path_dict["esa-energies"], map_ds["esa_energy_step"].data
    )
    esa_energy = energy_df.to_xarray()["nominal_central_energy"]

    # Convert ENA Signal Rate to Flux
    flux_conversion_divisor = geometric_factor * esa_energy
    map_ds["ena_intensity"] = map_ds["ena_signal_rates"] / flux_conversion_divisor
    map_ds["ena_intensity_stat_uncert"] = (
        map_ds["ena_signal_rate_stat_unc"] / flux_conversion_divisor
    )
    map_ds["ena_intensity_sys_err"] = (
        np.sqrt(map_ds["bg_rates"] * map_ds["exposure_factor"])
        / map_ds["exposure_factor"]
        / flux_conversion_divisor
    )

    # Combine calibration products using proper weighted averaging
    # as described in Hi Algorithm Document Section 3.1.2
    map_ds = combine_calibration_products(
        map_ds,
        geometric_factor,
        esa_energy,
    )

    if "raw" not in descriptor.principal_data:
        # Flux correction
        corrector = PowerLawFluxCorrector(l2_ancillary_path_dict["esa-eta-fit-factors"])
        # Apply flux correction with xarray inputs
        map_ds["ena_intensity"], map_ds["ena_intensity_stat_uncert"] = (
            corrector.apply_flux_correction(
                map_ds["ena_intensity"],
                map_ds["ena_intensity_stat_uncert"],
                esa_energy,
            )
        )

    return map_ds


def combine_calibration_products(
    map_ds: xr.Dataset,
    geometric_factors: xr.DataArray,
    esa_energies: xr.DataArray,
) -> xr.Dataset:
    """
    Combine calibration products using weighted averaging.

    Implements the algorithm described in Hi Algorithm Document Section 3.1.2
    for properly combining data from multiple calibration products.

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset that has preliminary intensity variables computed for each
        calibration product.
    geometric_factors : xarray.DataArray
        Geometric factors for each calibration product and energy step.
    esa_energies : xarray.DataArray
        Central energies for each energy step.

    Returns
    -------
    map_ds : xarray.Dataset
        Map dataset with updated variables: ena_intensity, ena_intensity_stat_uncert,
        ena_intensity_sys_err now combined across calibration products at each
        energy level.
    """
    ena_flux = map_ds["ena_intensity"]
    sys_err = map_ds["ena_intensity_sys_err"]

    # Calculate improved statistical variance estimates using geometric factor
    # ratios to reduce bias from Poisson uncertainty estimation
    improved_stat_variance = _calculate_improved_stat_variance(
        map_ds, geometric_factors, esa_energies
    )

    # Perform inverse-variance weighted averaging
    # Handle divide by zero and invalid values
    with np.errstate(divide="ignore", invalid="ignore"):
        # Use total variance weights for flux combination
        flux_weights = 1.0 / improved_stat_variance
        weighted_flux_sum = (ena_flux * flux_weights).sum(dim="calibration_prod")
        combined_flux = weighted_flux_sum / flux_weights.sum(dim="calibration_prod")

    map_ds["ena_intensity"] = combined_flux
    # Statistical uncertainty
    map_ds["ena_intensity_stat_uncert"] = np.sqrt(
        1 / (1 / (map_ds["ena_intensity_stat_uncert"] ** 2)).sum(dim="calibration_prod")
    )
    # For systematic error, just do quadrature sum over the systematic error for
    # each calibration product.
    map_ds["ena_intensity_sys_err"] = np.sqrt((sys_err**2).sum(dim="calibration_prod"))

    return map_ds


def _calculate_improved_stat_variance(
    map_ds: xr.Dataset,
    geometric_factors: xr.DataArray,
    esa_energies: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate improved statistical variances using geometric factor ratios.

    This implements the algorithm from Hi Algorithm Document Section 3.1.2:
    For calibration product X, replace N_X in the uncertainty calculation with
    an improved estimate using geometric factor ratios from all calibration products.

    The key insight is that we can vectorize this by first computing a geometric
    factor normalized signal rate, then scaling it back for each calibration product.

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset.
    geometric_factors : xr.DataArray
        Geometric factors for each calibration product.
    esa_energies : xarray.DataArray
        Central energies for each energy step.

    Returns
    -------
    improved_variance : xr.DataArray
        Improved statistical variance estimates.
    """
    n_calib_prods = map_ds["ena_intensity"].sizes.get("calibration_prod", 1)

    if n_calib_prods <= 1:
        # No improvement possible with single calibration product
        return map_ds["ena_intensity_stat_uncert"] ** 2

    logger.debug("Computing geometric factor normalized signal rates")

    # signal_rates = counts / exposure_factor - bg_rates
    # signal_rates shape is: (n_epoch, n_energy, n_cal_prod, n_spatial_pixels)
    signal_rates = map_ds["ena_signal_rates"]

    # Compute geometric factor normalized signal rate (vectorized approach)
    # This represents the weighted average signal rate per unit geometric factor
    # geometric_factor_norm_signal_rates shape is: (n_epoch, n_energy, n_spatial_pixels)
    geometric_factor_norm_signal_rates = signal_rates.sum(
        dim="calibration_prod"
    ) / geometric_factors.sum(dim="calibration_prod")

    # For each calibration product, the averaged signal rate estimate is:
    # averaged_signal_rate_i = geometric_factor_norm_signal_rates * geometric_factor_i
    # averaged_signal_rates shape is: (n_epoch, n_energy, n_cal_prod, n_spatial_pixels)
    averaged_signal_rates = geometric_factor_norm_signal_rates * geometric_factors

    logger.debug("Including background rates in uncertainty calculation")
    # Convert averaged signal rates back to flux uncertainties
    # Total count rates for Poisson uncertainty calculation
    total_count_rates_for_uncertainty = map_ds["bg_rates"] + averaged_signal_rates

    logger.debug("Computing improved flux uncertainties")
    # Statistical variance:
    with np.errstate(divide="ignore", invalid="ignore"):
        improved_variance = total_count_rates_for_uncertainty / (
            map_ds["exposure_factor"] * (geometric_factors * esa_energies)
        )

    # Handle invalid cases by falling back to original uncertainties
    improved_variance = xr.where(
        ~np.isfinite(improved_variance) | (geometric_factors == 0),
        map_ds["ena_intensity_stat_uncert"],
        improved_variance,
    )

    return improved_variance


# =============================================================================
# SETUP AND INITIALIZATION HELPERS
# =============================================================================


def esa_energy_df(
    esa_energies_path: str | Path, esa_energy_steps: np.ndarray | slice | None = None
) -> pd.DataFrame:
    """
    Lookup the nominal central energy values for given esa energy steps.

    Parameters
    ----------
    esa_energies_path : str or pathlib.Path
        Location of the calibration csv file containing the lookup data.
    esa_energy_steps : numpy.ndarray, slice, or None
        The ESA energy steps to get energies for. If not provided (default is None),
        the full dataframe is returned.

    Returns
    -------
    esa_energies_df: pandas.DataFrame
        Full data frame from the csv file filtered to only include the
        esa_energy_steps input.
    """
    if esa_energy_steps is None:
        esa_energy_steps = slice(None)
    esa_energies_lut = pd.read_csv(
        esa_energies_path, comment="#", index_col="esa_energy_step"
    )
    return esa_energies_lut.loc[esa_energy_steps]
