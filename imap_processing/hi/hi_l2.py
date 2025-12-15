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
    vars_to_bin = (
        HELIO_FRAME_VARS_TO_PROJECT
        if descriptor.frame_descriptor == "hf"
        else SC_FRAME_VARS_TO_PROJECT
    )
    vars_to_exposure_time_average = FULL_EXPOSURE_TIME_AVERAGE_SET & vars_to_bin

    if not isinstance(output_map, RectangularSkyMap):
        raise NotImplementedError("Healpix map output not supported for Hi")

    cached_esa_steps = None

    for pset_path in psets:
        logger.info(f"Processing {pset_path}")
        pset_ds = load_cdf(pset_path)

        # Rename some PSET vars to match L2 variables
        pset_ds = pset_ds.rename(HiPointingSet.l1c_to_l2_var_mapping)

        # Add obs_date variable to be used in determining a map mean obs_date
        mid_time = pset_ds["epoch"].values[0] + pset_ds["epoch_delta"].values[0] / 2
        pset_ds["obs_date"] = xr.full_like(pset_ds["exposure_factor"], float(mid_time))

        # Store the first PSET esa_energy_step values and make sure every PSET
        # contains the same set of esa_energy_step values.
        # TODO: Correctly handle PSETs with different esa_energy_step values.
        if cached_esa_steps is None:
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

        pset_ds = add_spacecraft_velocity_to_pset(pset_ds)

        if descriptor.frame_descriptor == "hf":
            # convert esa nominal central energy from keV to eV
            esa_energy_ev = energy_kev * 1000
            pset_ds = apply_compton_getting_correction(pset_ds, esa_energy_ev)

        pset_ds = calculate_ram_mask(pset_ds)

        # Multiply variables that need to be exposure time weighted average by
        # exposure factor.
        for var in vars_to_exposure_time_average:
            if var in pset_ds:
                pset_ds[var] *= pset_ds["exposure_factor"]

        # Project (bin) the PSET variables into the map pixels
        directional_mask = get_pset_directional_mask(pset_ds, descriptor.spin_phase)
        hi_pset = HiPointingSet(pset_ds)
        output_map.project_pset_values_to_map(
            hi_pset, list(vars_to_bin), pset_valid_mask=directional_mask
        )

    # Finish the exposure time weighted mean calculation of backgrounds
    # Allow divide by zero to fill set pixels with zero exposure time to NaN
    with np.errstate(divide="ignore"):
        for var in vars_to_exposure_time_average:
            output_map.data_1d[var] /= output_map.data_1d["exposure_factor"]

    output_map.data_1d.update(calculate_ena_signal_rates(output_map.data_1d))
    output_map.data_1d = calculate_ena_intensity(
        output_map.data_1d, l2_ancillary_path_dict, descriptor
    )

    # TODO: Handle variable types correctly in RectangularSkyMap.build_cdf_dataset
    output_map.data_1d["obs_date"].values = np.where(
        np.isfinite(output_map.data_1d["obs_date"].values),
        output_map.data_1d["obs_date"].values.astype(np.int64),
        np.int64(-9223372036854775808),
    )
    # TODO: Figure out how to compute obs_date_range (stddev of obs_date)
    output_map.data_1d["obs_date_range"] = xr.zeros_like(output_map.data_1d["obs_date"])

    # Set the energy_step_delta values to the energy bandpass half-width-half-max
    energy_delta = esa_ds["bandpass_fwhm"] / 2
    output_map.data_1d["energy_delta_minus"] = energy_delta
    output_map.data_1d["energy_delta_plus"] = energy_delta

    # Rename and convert coordinate from esa_energy_step energy
    output_map.data_1d = output_map.data_1d.rename({"esa_energy_step": "energy"})
    output_map.data_1d = output_map.data_1d.assign_coords(energy=energy_kev.values)

    output_map.data_1d = output_map.data_1d.drop("esa_energy_step_label")

    # Apply Compton-Getting interpolation for heliocentric frame maps
    if descriptor.frame_descriptor == "hf":
        esa_energy_ev = esa_energy_ev.rename({"esa_energy_step": "energy"})
        esa_energy_ev = esa_energy_ev.assign_coords(energy=energy_kev.values)
        output_map.data_1d = interpolate_map_flux_to_helio_frame(
            output_map.data_1d,
            output_map.data_1d["energy"] * 1000,  # Convert ESA energies to eV
            esa_energy_ev,  # heliocentric energies (same as ESA energies)
            ["ena_intensity"],
        )

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
