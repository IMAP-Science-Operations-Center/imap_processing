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
    add_spacecraft_position_and_velocity_to_pset,
    apply_compton_getting_correction,
    calculate_ram_mask,
    get_pset_directional_mask,
    interpolate_map_flux_to_helio_frame,
)
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.hi.utils import CalibrationProductConfig

logger = logging.getLogger(__name__)

SC_FRAME_VARS_TO_PROJECT = {
    "ena_count",
    "exposure_factor",
    "bg_rate",
    "bg_rate_sys_err",
    "obs_date",
}
HELIO_FRAME_VARS_TO_PROJECT = SC_FRAME_VARS_TO_PROJECT | {"energy_sc"}
# TODO: is an exposure time weighted average for obs_date appropriate?
FULL_EXPOSURE_TIME_AVERAGE_SET = {"bg_rate", "bg_rate_sys_err", "obs_date", "energy_sc"}

# Calibration systematic uncertainty as a fraction of intensity.
# This represents the current calibration uncertainty (40%).
CALIBRATION_UNCERTAINTY_FRACTION = 0.40


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
    sky_maps = create_sky_map_from_psets(
        psets,
        l2_ancillary_path_dict,
        map_descriptor,
    )

    logger.info("Step 2: Calculating rates and intensities")
    for sky_map in sky_maps.values():
        sky_map.data_1d = calculate_all_rates_and_intensities(
            sky_map.data_1d,
            l2_ancillary_path_dict,
            map_descriptor,
        )

    logger.info("Step 3: Combining maps (if needed)")
    final_map = combine_maps(sky_maps)

    logger.info("Step 4: Finalizing dataset with attributes")
    l2_ds = final_map.build_cdf_dataset(
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
) -> dict[str, RectangularSkyMap]:
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
    sky_maps : dict[str, RectangularSkyMap]
        Dictionary mapping spin phase keys ("ram", "anti", or "full")
        to sky maps with all the PSET data projected. Includes
        an energy coordinate and energy_delta_minus and energy_delta_plus
        variables from ESA energy calibration data. For helioframe full-spin
        maps, contains "ram" and "anti" keys; otherwise contains a single key
        matching the descriptor's spin_phase.
    """
    if len(psets) == 0:
        raise ValueError("No PSETs provided for map creation")

    # If we are making a full-spin, helio-frame map, we need to make one ram
    # and one anti-ram map that get combined at the final L2 step.
    if descriptor.frame_descriptor == "hf" and descriptor.spin_phase == "full":
        # The spin-phase of the descriptor has no effect on the output map, so
        # we can just use descriptor.to_empty_map() to generate both.
        output_maps = {
            "ram": descriptor.to_empty_map(),
            "anti": descriptor.to_empty_map(),
        }
    else:
        output_maps = {descriptor.spin_phase: descriptor.to_empty_map()}

    if not all([isinstance(map, RectangularSkyMap) for map in output_maps.values()]):
        raise NotImplementedError("Healpix map output not supported for Hi")
    # Needed for mypy type narrowing
    rect_maps: dict[str, RectangularSkyMap] = {
        k: v for k, v in output_maps.items() if isinstance(v, RectangularSkyMap)
    }

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

        hi_pset = HiPointingSet(pset_processed)

        for spin_phase, map in rect_maps.items():
            # Project (bin) the PSET variables into the map pixels
            directional_mask = get_pset_directional_mask(pset_processed, spin_phase)
            map.project_pset_values_to_map(
                hi_pset, list(vars_to_bin), pset_valid_mask=directional_mask
            )

    for map in rect_maps.values():
        # Finish the exposure time weighted mean calculation of backgrounds
        # Allow divide by zero to fill set pixels with zero exposure time to NaN
        with np.errstate(divide="ignore"):
            map.data_1d[vars_to_exposure_time_average] /= map.data_1d["exposure_factor"]

        # Add ESA energy data to the map dataset for use in rate/intensity calculations
        energy_delta = esa_ds["bandpass_fwhm"] / 2
        map.data_1d["energy_delta_minus"] = energy_delta
        map.data_1d["energy_delta_plus"] = energy_delta
        # Add energy as an auxiliary coordinate (keV values indexed by esa_energy_step)
        map.data_1d = map.data_1d.assign_coords(
            energy=("esa_energy_step", esa_ds["nominal_central_energy"].values)
        )

    return rect_maps


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

    # Step 3: Add spacecraft position and velocity
    pset_processed = add_spacecraft_position_and_velocity_to_pset(pset_processed)

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
        Map dataset with projected PSET data (ena_count, exposure_factor, bg_rate,
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

    # Step 3: Add calibration systematic uncertainty in quadrature with the
    # background-associated systematic. This is a percentage of the intensity.
    # calib_sys_err is computed from ena_intensity after flux correction has
    # already been applied (see calculate_ena_intensity), so it is automatically
    # consistent with the corrected intensity and needs no separate scaling.
    # This must happen before the CG interpolation step below, which requires
    # ena_intensity_sys_err to already exist (even though it deliberately
    # leaves it unmodified -- see update_sys_err=False).
    # For full-spin (ram+anti) maps, these per-map values are provisional:
    # combine_maps() recalculates both from the combined ena_intensity and
    # bg_intensity_sys_err rather than exposure-weight-averaging these.
    logger.debug("Adding calibration systematic uncertainty")
    bg_sys_err = map_ds["bg_intensity_sys_err"]
    calib_sys_err = CALIBRATION_UNCERTAINTY_FRACTION * map_ds["ena_intensity"]
    map_ds["ena_intensity_calibration_sys_err"] = calib_sys_err
    map_ds["ena_intensity_sys_err"] = np.sqrt(bg_sys_err**2 + calib_sys_err**2)

    # Step 4: Handle obs_date variable type conversion
    # TODO: Handle variable types correctly in RectangularSkyMap.build_cdf_dataset
    obs_date = map_ds["obs_date"]
    # Replace non-finite values with the int64 sentinel before casting
    map_ds["obs_date"] = xr.where(
        np.isfinite(obs_date),
        obs_date.astype("int64"),
        np.int64(-9223372036854775808),
    )
    map_ds["obs_date_range"] = xr.zeros_like(map_ds["obs_date"])

    # Step 5: Swap esa_energy_step dimension for energy coordinate
    map_ds = map_ds.swap_dims({"esa_energy_step": "energy"})
    map_ds = map_ds.drop_vars(
        ["esa_energy_step", "esa_energy_step_label"], errors="ignore"
    )

    # Step 6: Apply Compton-Getting interpolation for heliocentric frame maps
    if descriptor.frame_descriptor == "hf":
        logger.debug("Applying Compton-Getting interpolation for heliocentric frame")
        # Convert energy coordinate from keV to eV for interpolation
        esa_energy_ev = map_ds["energy"] * 1000

        # Hi does not want to apply the flux correction to the systematic error.
        map_ds = interpolate_map_flux_to_helio_frame(
            map_ds,
            esa_energy_ev,  # ESA energies in eV
            esa_energy_ev,  # heliocentric energies (same as ESA energies)
            ["ena_intensity"],
            update_sys_err=False,  # Hi does not update the systematic error
        )
        # Drop any esa_energy_step_label that may have been re-added
        map_ds = map_ds.drop_vars(["esa_energy_step_label"], errors="ignore")

    # Step 7: Clean up intermediate variables
    map_ds = cleanup_intermediate_variables(map_ds)

    return map_ds


def calculate_ena_signal_rates(map_ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the ENA signal rates.

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset that has ena_count, exposure_factor, and bg_rate calculated.

    Returns
    -------
    map_ds : xarray.Dataset
        Map dataset with new variables: ena_signal_rates, ena_signal_rate_stat_unc.
    """
    # Allow divide by zero to set pixels with zero exposure time to NaN
    with np.errstate(divide="ignore"):
        # Calculate the ENA Signal Rate
        map_ds["ena_signal_rates"] = (
            map_ds["ena_count"] / map_ds["exposure_factor"] - map_ds["bg_rate"]
        )
        # Calculate the ENA Signal Rate Uncertainties
        # The minimum count uncertainty is 1 for any pixel that has non-zero
        # exposure time. See IMAP Hi Algorithm Document section 3.1.1. Here,
        # we can ignore the non-zero exposure time condition when setting the
        # minimum count uncertainty because division by zero exposure time results
        # in the correct NaN value.
        min_counts_unc = xr.ufuncs.maximum(map_ds["ena_count"], 1)
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
        bg_intensity_sys_err.
    """
    # read calibration product configuration file
    cal_prod_df = CalibrationProductConfig.from_csv(l2_ancillary_path_dict["cal-prod"])
    # L2 does not yet combine PSETs from different gain states into a single
    # map (see hi_l1c.add_pset_geometric_factor()'s docstring), so use the
    # first (and, today, only) gain_config_id present in the ancillary file.
    gain_config_ids = cal_prod_df.index.get_level_values("gain_config_id").unique()
    if len(gain_config_ids) != 1:
        raise NotImplementedError(
            "L2 processing does not yet support multiple gain_config_id values."
        )
    cal_prod_df = cal_prod_df.loc[gain_config_ids[0]]
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

    # Convert the exposure-time weighted average of background rate systematic
    # uncertainty from rate to intensity units.
    map_ds["bg_intensity_sys_err"] = map_ds["bg_rate_sys_err"] / flux_conversion_divisor

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
        pre_correction_intensity = map_ds["ena_intensity"]
        map_ds["ena_intensity"], map_ds["ena_intensity_stat_uncert"] = (
            corrector.apply_flux_correction(
                map_ds["ena_intensity"],
                map_ds["ena_intensity_stat_uncert"],
                esa_energy,
            )
        )
        # Scale the background systematic error by the same flux correction
        # ratio that was just applied to ena_intensity.
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_correction_ratio = map_ds["ena_intensity"] / pre_correction_intensity
        map_ds["bg_intensity_sys_err"] = (
            map_ds["bg_intensity_sys_err"] * flux_correction_ratio
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
        bg_intensity_sys_err, ena_count, bg_rate,
        bg_rate_sys_err now combined across calibration products at each
        energy level.
    """
    ena_flux = map_ds["ena_intensity"]
    sys_err = map_ds["bg_intensity_sys_err"]

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
        1
        / (1 / (map_ds["ena_intensity_stat_uncert"] ** 2)).sum(
            dim="calibration_prod", skipna=True, min_count=1
        )
    )
    # For systematic error, just do quadrature sum over the systematic error for
    # each calibration product.
    map_ds["bg_intensity_sys_err"] = np.sqrt(
        (sys_err**2).sum(dim="calibration_prod", skipna=True, min_count=1)
    )

    # ena_count is a diagnostic tally of binned direct events, so it is simply
    # summed across calibration products.
    map_ds["ena_count"] = map_ds["ena_count"].sum(dim="calibration_prod")

    # bg_rate is combined the same way as ena_intensity_stat_uncert above:
    # inverse-variance weighted using each calibration product's own
    # bg_rate_sys_err as its uncertainty.
    with np.errstate(divide="ignore", invalid="ignore"):
        bg_rate_weights = xr.where(
            map_ds["bg_rate_sys_err"] > 0,
            1.0 / (map_ds["bg_rate_sys_err"] ** 2),
            0.0,
        )
        weight_sum = bg_rate_weights.sum(
            dim="calibration_prod", skipna=True, min_count=1
        )
        weighted_bg_sum = (map_ds["bg_rate"] * bg_rate_weights).sum(
            dim="calibration_prod", skipna=True, min_count=1
        )
        map_ds["bg_rate"] = xr.where(
            weight_sum > 0, weighted_bg_sum / weight_sum, np.nan
        )
        map_ds["bg_rate_sys_err"] = xr.where(
            weight_sum > 0, np.sqrt(1 / weight_sum), np.nan
        )

    return map_ds


def combine_maps(sky_maps: dict[str, RectangularSkyMap]) -> RectangularSkyMap:
    """
    Combine ram and anti-ram sky maps using appropriate weighting.

    For full-spin heliocentric frame maps, ram and anti-ram maps are processed
    separately through the CG correction pipeline, then combined here using
    inverse-variance weighting for intensity and appropriate methods for other
    variables.

    Parameters
    ----------
    sky_maps : dict[str, RectangularSkyMap]
        Dictionary of sky maps to combine. Expected to contain either 1 map
        (no combination needed) or 2 maps with "ram" and "anti" keys.

    Returns
    -------
    RectangularSkyMap
        Combined sky map.
    """
    if len(sky_maps) not in [1, 2]:
        raise ValueError(f"Expected 1 or 2 sky maps, got {len(sky_maps)}")
    if len(sky_maps) == 1:
        logger.debug("Only one sky map provided, returning it unchanged")
        return next(iter(sky_maps.values()))
    if "ram" not in sky_maps or "anti" not in sky_maps:
        raise ValueError(
            f"Expected sky maps with 'ram' and 'anti' keys."
            f"Instead got: {sky_maps.keys()}"
        )

    logger.info("Combining ram and anti-ram maps using inverse-variance weighting.")

    ram_ds = sky_maps["ram"].data_1d
    anti_ds = sky_maps["anti"].data_1d

    # Use the ram sky map as the base for the combined result
    combined_map = sky_maps["ram"]
    combined = ram_ds.copy()

    # Additive variables: ena_count and exposure_factor
    combined["ena_count"] = ram_ds["ena_count"] + anti_ds["ena_count"]
    combined["exposure_factor"] = ram_ds["exposure_factor"] + anti_ds["exposure_factor"]

    # Compute weights for ram and anti-ram based on the inverse of the variance
    # (uncertainty squared). Zero weights where the variance is zero or where
    # ena_intensity is not finite. The latter prevents NaNs, which get replaced
    # with zeros for the sum below, from contributing to the weighted average.
    ram_valid_mask = np.logical_and(
        ram_ds["ena_intensity_stat_uncert"] > 0, np.isfinite(ram_ds["ena_intensity"])
    )
    weight_ram = xr.where(
        ram_valid_mask,
        1 / ram_ds["ena_intensity_stat_uncert"] ** 2,
        0,
    )
    anti_valid_mask = np.logical_and(
        anti_ds["ena_intensity_stat_uncert"] > 0, np.isfinite(anti_ds["ena_intensity"])
    )
    weight_anti = xr.where(
        anti_valid_mask,
        1 / anti_ds["ena_intensity_stat_uncert"] ** 2,
        0,
    )
    total_weight = weight_ram + weight_anti

    with np.errstate(divide="ignore", invalid="ignore"):
        # Inverse-variance weighted average for ena_intensity
        combined["ena_intensity"] = (
            ram_ds["ena_intensity"].fillna(0) * weight_ram
            + anti_ds["ena_intensity"].fillna(0) * weight_anti
        ) / total_weight
        # ena_intensity_stat_uncertainty is combined using inverse quadrature sum
        combined["ena_intensity_stat_uncert"] = np.sqrt(1 / total_weight)

    # Exposure-weighted average for background rate and systematic error
    # variables.
    # NaNs in these variables should occur only where the exposure_factor
    # is zero. This means the correct NaN handling is to just replace NaNs in
    # these variables with zeros so that the sum is not affected.
    with np.errstate(divide="ignore", invalid="ignore"):
        total_exp = combined["exposure_factor"]
        for var in (
            "bg_rate",
            "bg_rate_sys_err",
            "bg_intensity_sys_err",
        ):
            combined[var] = (
                ram_ds[var].fillna(0) * ram_ds["exposure_factor"]
                + anti_ds[var].fillna(0) * anti_ds["exposure_factor"]
            ) / total_exp

    # ena_intensity_calibration_sys_err and ena_intensity_sys_err are
    # recalculated from the combined map's own values rather than
    # exposure-weight-averaged from the ram/anti maps' pre-combination values.
    # ena_intensity_calibration_sys_err is a fixed percentage of the combined
    # ena_intensity, and ena_intensity_sys_err is the quadrature sum of that
    # and the (now combined) background intensity systematic error.
    combined["ena_intensity_calibration_sys_err"] = (
        CALIBRATION_UNCERTAINTY_FRACTION * combined["ena_intensity"]
    )
    combined["ena_intensity_sys_err"] = np.sqrt(
        combined["bg_intensity_sys_err"] ** 2
        + combined["ena_intensity_calibration_sys_err"] ** 2
    )

    # Exposure-weighted average for obs_date
    with np.errstate(divide="ignore", invalid="ignore"):
        combined["obs_date"] = (
            ram_ds["obs_date"] * ram_ds["exposure_factor"]
            + anti_ds["obs_date"] * anti_ds["exposure_factor"]
        ) / total_exp

        # Combined obs_date_range accounts for within-group and between-group variance
        # var_combined = (w1*var1 + w2*var2)/(w1+w2) + w1*w2*(mean1-mean2)^2/(w1+w2)^2
        within_variance = (
            ram_ds["exposure_factor"] * ram_ds["obs_date_range"] ** 2
            + anti_ds["exposure_factor"] * anti_ds["obs_date_range"] ** 2
        ) / total_exp
        between_variance = (
            ram_ds["exposure_factor"]
            * anti_ds["exposure_factor"]
            * (ram_ds["obs_date"] - anti_ds["obs_date"]) ** 2
        ) / (total_exp**2)
        combined["obs_date_range"] = np.sqrt(within_variance + between_variance)

    # Re-cast obs_date and obs_date_range back to int64 after float arithmetic.
    # Replace non-finite values with the int64 sentinel value.
    # TODO: Handle variable types correctly in RectangularSkyMap.build_cdf_dataset
    int64_sentinel = np.int64(-9223372036854775808)
    combined["obs_date"] = xr.where(
        np.isfinite(combined["obs_date"]),
        combined["obs_date"].astype("int64"),
        int64_sentinel,
    )
    combined["obs_date_range"] = xr.where(
        np.isfinite(combined["obs_date_range"]),
        combined["obs_date_range"].astype("int64"),
        int64_sentinel,
    )

    combined_map.data_1d = combined
    return combined_map


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

    # signal_rates = ena_count / exposure_factor - bg_rate
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
    total_count_rates_for_uncertainty = map_ds["bg_rate"] + averaged_signal_rates

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


def cleanup_intermediate_variables(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remove intermediate variables that were only needed for calculations.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing intermediate calculation variables.

    Returns
    -------
    xarray.Dataset
        Cleaned dataset with intermediate variables removed.
    """
    # Remove the intermediate variables from the map
    potential_vars = [
        "energy_sc",
        "ena_signal_rates",
        "ena_signal_rate_stat_unc",
    ]

    vars_to_remove = [var for var in potential_vars if var in dataset.data_vars]

    return dataset.drop_vars(vars_to_remove)


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
