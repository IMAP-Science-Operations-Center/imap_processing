"""IMAP-HI L2 processing module."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.ena_maps.ena_maps import (
    HiPointingSet,
    RectangularSkyMap,
)
from imap_processing.ena_maps.utils.corrections import PowerLawFluxCorrector
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.hi.utils import CalibrationProductConfig

logger = logging.getLogger(__name__)

# TODO: is an exposure time weighted average for obs_date appropriate?
VARS_TO_EXPOSURE_TIME_AVERAGE = ["bg_rates", "bg_rates_unc", "obs_date"]


def hi_l2(
    psets: list[str | Path],
    l2_ancillary_path_dict: dict[str, Path],
    descriptor: str,
) -> list[xr.Dataset]:
    """
    High level IMAP-Hi L2 processing function.

    Parameters
    ----------
    psets : list of str or pathlib.Path
        List of input PSETs to make a map from.
    l2_ancillary_path_dict : dict[str, pathlib.Path]
        Mapping containing ancillary file descriptors as keys and file paths as
        values. Require keys are: ["cal-prod", "esa-energies", "esa-eta-fit-factors"].
    descriptor : str
        Output filename descriptor. Contains full configuration for the options
        of how to generate the map.

    Returns
    -------
    l2_dataset : list[xarray.Dataset]
        Level 2 IMAP-Hi dataset ready to be written to a CDF file.
    """
    logger.info(
        f"Hi L2 processing running for descriptor: {descriptor} with"
        f"{len(psets)} PSETs input."
    )

    map_descriptor = MapDescriptor.from_string(descriptor)
    if not isinstance(map_descriptor.sensor, str):
        raise ValueError(
            "Invalid map_descriptor. Sensor attribute must be of type str "
            "and be either '45' or '90'"
        )

    sky_map = generate_hi_map(
        psets,
        l2_ancillary_path_dict,
        map_descriptor,
    )

    l2_ds = sky_map.build_cdf_dataset(
        "hi",
        "l2",
        descriptor,
        sensor=map_descriptor.sensor,
    )

    return [l2_ds]


def generate_hi_map(
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
        The sky map with all the PSET data projected into the map.
    """
    output_map = descriptor.to_empty_map()

    if not isinstance(output_map, RectangularSkyMap):
        raise NotImplementedError("Healpix map output not supported for Hi")

    # TODO: Implement Compton-Getting correction
    if descriptor.frame_descriptor != "sf":
        raise NotImplementedError("CG correction not implemented for Hi")

    for pset_path in psets:
        logger.info(f"Processing {pset_path}")
        pset = HiPointingSet(pset_path, spin_phase=descriptor.spin_phase)

        # Background rate and uncertainty are exposure time weighted means in
        # the map.
        for var in VARS_TO_EXPOSURE_TIME_AVERAGE:
            pset.data[var] *= pset.data["exposure_factor"]

        # Project (bin) the PSET variables into the map pixels
        output_map.project_pset_values_to_map(
            pset,
            ["counts", "exposure_factor", "bg_rates", "bg_rates_unc", "obs_date"],
        )

    # Finish the exposure time weighted mean calculation of backgrounds
    # Allow divide by zero to fill set pixels with zero exposure time to NaN
    with np.errstate(divide="ignore"):
        for var in VARS_TO_EXPOSURE_TIME_AVERAGE:
            output_map.data_1d[var] /= output_map.data_1d["exposure_factor"]

    output_map.data_1d.update(calculate_ena_signal_rates(output_map.data_1d))
    output_map.data_1d = calculate_ena_intensity(
        output_map.data_1d, l2_ancillary_path_dict, descriptor
    )

    output_map.data_1d["obs_date"].data = output_map.data_1d["obs_date"].data.astype(
        np.int64
    )
    # TODO: Figure out how to compute obs_date_range (stddev of obs_date)
    output_map.data_1d["obs_date_range"] = xr.zeros_like(output_map.data_1d["obs_date"])

    # Rename and convert coordinate from esa_energy_step energy
    esa_df = esa_energy_df(
        l2_ancillary_path_dict["esa-energies"],
        output_map.data_1d["esa_energy_step"].data,
    )
    output_map.data_1d = output_map.data_1d.rename({"esa_energy_step": "energy"})
    output_map.data_1d = output_map.data_1d.assign_coords(
        energy=esa_df["nominal_central_energy"].values
    )
    # Set the energy_step_delta values to the energy bandpass half-width-half-max
    energy_delta = esa_df["bandpass_fwhm"].values / 2
    output_map.data_1d["energy_delta_minus"] = xr.DataArray(
        energy_delta,
        name="energy_delta_minus",
        dims=["energy"],
    )
    output_map.data_1d["energy_delta_plus"] = xr.DataArray(
        energy_delta,
        name="energy_delta_plus",
        dims=["energy"],
    )

    output_map.data_1d = output_map.data_1d.drop("esa_energy_step_label")

    return output_map


def calculate_ena_signal_rates(map_ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """
    Calculate the ENA signal rates.

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset that has counts, exposure_times, and background_rates calculated.

    Returns
    -------
    signal_rates : dict[str, xarray.DataArray]
        ENA signal rates computed from the binned PSET data.
    """
    signal_rate_vars = {}
    # Allow divide by zero to set pixels with zero exposure time to NaN
    with np.errstate(divide="ignore"):
        # Calculate the ENA Signal Rate
        signal_rate_vars["ena_signal_rates"] = (
            map_ds["counts"] / map_ds["exposure_factor"] - map_ds["bg_rates"]
        )
        # Calculate the ENA Signal Rate Uncertainties
        # The minimum count uncertainty is 1 for any pixel that has non-zero
        # exposure time. See IMAP Hi Algorithm Document section 3.1.1. Here,
        # we can ignore the non-zero exposure time condition when setting the
        # minimum count uncertainty because division by zero exposure time results
        # in the correct NaN value.
        min_counts_unc = xr.ufuncs.maximum(map_ds["counts"], 1)
        signal_rate_vars["ena_signal_rate_stat_unc"] = (
            np.sqrt(min_counts_unc) / map_ds["exposure_factor"]
        )

    # Statistical fluctuations may result in a negative ENA signal rate after
    # background subtraction. A negative signal rate is nonphysical. See IMAP Hi
    # Algorithm Document section 3.1.1
    signal_rate_vars["ena_signal_rates"].values[
        signal_rate_vars["ena_signal_rates"].values < 0
    ] = 0
    return signal_rate_vars


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
        Map dataset with new variables: ena_intensity, ena_intensity_stat_unc,
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
    map_ds["ena_intensity_stat_unc"] = (
        map_ds["ena_signal_rate_stat_unc"] / flux_conversion_divisor
    )
    map_ds["ena_intensity_sys_err"] = map_ds["bg_rates_unc"] / flux_conversion_divisor

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
        # FluxCorrector does not accept the size 1 epoch dimension. Remove that
        # dimension by passing the zeroth element.
        corrected_intensity, corrected_stat_unc = corrector.apply_flux_correction(
            map_ds["ena_intensity"].values[0],
            map_ds["ena_intensity_stat_unc"].values[0],
            esa_energy.data,
        )
        # Add the size 1 epoch dimension back in to the corrected fluxes.
        map_ds["ena_intensity"].data = corrected_intensity[np.newaxis, ...]
        map_ds["ena_intensity_stat_unc"].data = corrected_stat_unc[np.newaxis, ...]

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
        Map dataset with updated variables: ena_intensity, ena_intensity_stat_unc,
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

    # Calculate total variance
    # Note that sys_err contains uncertainty, so it must be squared to get
    # the systematic variance needed in this equation.
    total_variance = improved_stat_variance + sys_err**2

    # Perform inverse-variance weighted averaging
    # Handle divide by zero and invalid values
    with np.errstate(divide="ignore", invalid="ignore"):
        # Calculate weights for statistical variance combination using only
        # statistical variance
        stat_weights = 1.0 / improved_stat_variance

        # Combined statistical uncertainty from inverse-variance formula
        combined_stat_unc = np.sqrt(1.0 / stat_weights.sum(dim="calibration_prod"))

        # Use total variance weights for flux combination
        flux_weights = 1.0 / total_variance
        weighted_flux_sum = (ena_flux * flux_weights).sum(dim="calibration_prod")
        combined_flux = weighted_flux_sum / flux_weights.sum(dim="calibration_prod")

    map_ds["ena_intensity"] = combined_flux
    map_ds["ena_intensity_stat_unc"] = combined_stat_unc
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
        return map_ds["ena_intensity_stat_unc"] ** 2

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

    # Ensure non-negative values for sqrt and minimum of 1 for uncertainty calculation
    total_count_rates_for_uncertainty = xr.where(
        total_count_rates_for_uncertainty < 1, 1, total_count_rates_for_uncertainty
    )

    logger.debug("Computing improved flux uncertainties")
    # Statistical variance:
    with np.errstate(divide="ignore", invalid="ignore"):
        improved_variance = total_count_rates_for_uncertainty / (
            map_ds["exposure_factor"] * (geometric_factors * esa_energies)
        )

    # Handle invalid cases by falling back to original uncertainties
    improved_variance = xr.where(
        ~np.isfinite(improved_variance) | (geometric_factors == 0),
        map_ds["ena_intensity_stat_unc"],
        improved_variance,
    )

    return improved_variance


def esa_energy_df(
    esa_energies_path: str | Path, esa_energy_steps: np.ndarray
) -> pd.DataFrame:
    """
    Lookup the nominal central energy values for given esa energy steps.

    Parameters
    ----------
    esa_energies_path : str or pathlib.Path
        Location of the calibration csv file containing the lookup data.
    esa_energy_steps : numpy.ndarray
        The ESA energy steps to get energies for.

    Returns
    -------
    esa_energies_df: pandas.DataFrame
        Full data frame from the csv file filtered to only include the
        esa_energy_steps input.
    """
    esa_energies_lut = pd.read_csv(
        esa_energies_path, comment="#", index_col="esa_energy_step"
    )
    return esa_energies_lut.loc[esa_energy_steps]
