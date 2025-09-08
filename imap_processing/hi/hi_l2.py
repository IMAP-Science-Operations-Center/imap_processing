"""IMAP-HI L2 processing module."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.ena_maps.ena_maps import (
    AbstractSkyMap,
    HiPointingSet,
    RectangularSkyMap,
)
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.hi.utils import CalibrationProductConfig

logger = logging.getLogger(__name__)

# TODO: is an exposure time weighted average for obs_date appropriate?
VARS_TO_EXPOSURE_TIME_AVERAGE = ["bg_rates", "bg_rates_unc", "obs_date"]


def hi_l2(
    psets: list[str | Path],
    geometric_factors_path: str | Path,
    esa_energies_path: str | Path,
    descriptor: str,
) -> list[xr.Dataset]:
    """
    High level IMAP-Hi L2 processing function.

    Parameters
    ----------
    psets : list of str or pathlib.Path
        List of input PSETs to make a map from.
    geometric_factors_path : str or pathlib.Path
        Where to get the geometric factors from.
    esa_energies_path : str or pathlib.Path
        Where to get the energies from.
    descriptor : str
        Output filename descriptor. Contains full configuration for the options
        of how to generate the map.

    Returns
    -------
    l2_dataset : list[xarray.Dataset]
        Level 2 IMAP-Hi dataset ready to be written to a CDF file.
    """
    cg_corrected = False
    map_descriptor = MapDescriptor.from_string(descriptor)

    sky_map = generate_hi_map(
        psets,
        geometric_factors_path,
        esa_energies_path,
        spin_phase=map_descriptor.spin_phase,
        output_map=map_descriptor.to_empty_map(),
        cg_corrected=cg_corrected,
    )

    # Get the map dataset with variables/coordinates in the correct shape
    # TODO get the correct descriptor and frame

    if not isinstance(sky_map, RectangularSkyMap):
        raise NotImplementedError("HEALPix map output not supported for Hi")
    if not isinstance(map_descriptor.sensor, str):
        raise ValueError(
            "Invalid map_descriptor. Sensor attribute must be of type str "
            "and be either '45' or '90'"
        )

    l2_ds = sky_map.build_cdf_dataset(
        "hi",
        "l2",
        map_descriptor.frame_descriptor,
        descriptor,
        sensor=map_descriptor.sensor,
    )

    return [l2_ds]


def generate_hi_map(
    psets: list[str | Path],
    geometric_factors_path: str | Path,
    esa_energies_path: str | Path,
    output_map: AbstractSkyMap,
    cg_corrected: bool = False,
    spin_phase: str = "full",
) -> AbstractSkyMap:
    """
    Project Hi PSET data into a sky map.

    Parameters
    ----------
    psets : list of str or pathlib.Path
        List of input PSETs to make a map from.
    geometric_factors_path : str or pathlib.Path
        Where to get the geometric factors from.
    esa_energies_path : str or pathlib.Path
        Where to get the energies from.
    output_map : AbstractSkyMap
        The map object to collect data into. Determines pixel spacing,
        coordinate system, etc.
    cg_corrected : bool, Optional
        Whether to apply Compton-Getting correction to the energies. Defaults to
        False.
    spin_phase : str, Optional
        Apply filtering to PSET data include ram or anti-ram or full spin data.
        Defaults to "full".

    Returns
    -------
    sky_map : AbstractSkyMap
        The sky map with all the PSET data projected into the map.
    """
    # TODO: Implement Compton-Getting correction
    if cg_corrected:
        raise NotImplementedError

    for pset_path in psets:
        logger.info(f"Processing {pset_path}")
        pset = HiPointingSet(pset_path, spin_phase=spin_phase)

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
    output_map.data_1d.update(
        calculate_ena_intensity(
            output_map.data_1d, geometric_factors_path, esa_energies_path
        )
    )

    output_map.data_1d["obs_date"].data = output_map.data_1d["obs_date"].data.astype(
        np.int64
    )
    # TODO: Figure out how to compute obs_date_range (stddev of obs_date)
    output_map.data_1d["obs_date_range"] = xr.zeros_like(output_map.data_1d["obs_date"])

    # Rename and convert coordinate from esa_energy_step energy
    esa_df = esa_energy_df(
        esa_energies_path, output_map.data_1d["esa_energy_step"].data
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
    geometric_factors_path: str | Path,
    esa_energies_path: str | Path,
) -> dict[str, xr.DataArray]:
    """
    Calculate the ena intensities.

    Parameters
    ----------
    map_ds : xarray.Dataset
        Map dataset that has ena_signal_rate fields calculated.
    geometric_factors_path : str or pathlib.Path
        Where to get the geometric factors from.
    esa_energies_path : str or pathlib.Path
        Where to get the esa energies, energy deltas, and geometric factors.

    Returns
    -------
    intensity_vars : dict[str, xarray.DataArray]
        ENA Intensity with statistical and systematic uncertainties.
    """
    # read calibration product configuration file
    cal_prod_df = CalibrationProductConfig.from_csv(geometric_factors_path)
    # reindex_like removes esa_energy_steps and calibration products not in the
    # map_ds esa_energy_step and calibration_product coordinates
    geometric_factor = cal_prod_df.to_xarray().reindex_like(map_ds)["geometric_factor"]
    geometric_factor = geometric_factor.transpose(
        *[coord for coord in map_ds.coords if coord in geometric_factor.coords]
    )
    energy_df = esa_energy_df(esa_energies_path, map_ds["esa_energy_step"].data)
    esa_energy = energy_df.to_xarray()["nominal_central_energy"]

    # Convert ENA Signal Rate to Flux
    flux_conversion_divisor = geometric_factor * esa_energy
    intensity_vars = {
        "ena_intensity": map_ds["ena_signal_rates"] / flux_conversion_divisor,
        "ena_intensity_stat_unc": map_ds["ena_signal_rate_stat_unc"]
        / flux_conversion_divisor,
        "ena_intensity_sys_err": map_ds["bg_rates_unc"] / flux_conversion_divisor,
    }

    # TODO: Correctly implement combining of calibration products. For now, just sum
    # Hi groups direct events into distinct calibration products based on coincidence
    # type. (See L1B processing and Hi Algorithm Document section 6.1.2) When adding
    # together different calibration products, a different weighting must be used
    # than exposure time. (See Hi Algorithm Document Section 3.1.2)
    intensity_vars["ena_intensity"] = intensity_vars["ena_intensity"].sum(
        dim="calibration_prod"
    )
    intensity_vars["ena_intensity_stat_unc"] = np.sqrt(
        (intensity_vars["ena_intensity_stat_unc"] ** 2).sum(dim="calibration_prod")
    )
    intensity_vars["ena_intensity_sys_err"] = np.sqrt(
        (intensity_vars["ena_intensity_sys_err"] ** 2).sum(dim="calibration_prod")
    )

    return intensity_vars


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
