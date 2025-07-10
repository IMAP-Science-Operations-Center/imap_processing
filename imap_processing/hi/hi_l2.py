"""IMAP-HI L2 processing module."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from imap_processing.ena_maps.ena_maps import HiPointingSet, RectangularSkyMap
from imap_processing.spice.geometry import SpiceFrame

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
    # TODO: parse descriptor to determine map configuration
    sensor = "45" if "45" in descriptor else "90"
    direction: Literal["full"] = "full"
    cg_corrected = False
    map_spacing = 4

    rect_map = generate_hi_map(
        psets,
        geometric_factors_path,
        esa_energies_path,
        direction=direction,
        cg_corrected=cg_corrected,
        map_spacing=map_spacing,
    )

    # Get the map dataset with variables/coordinates in the correct shape
    # TODO get the correct descriptor and frame
    l2_ds = rect_map.build_cdf_dataset("hi", "l2", "sf", descriptor, sensor=sensor)

    return [l2_ds]


def generate_hi_map(
    psets: list[str | Path],
    geometric_factors_path: str | Path,
    esa_energies_path: str | Path,
    cg_corrected: bool = False,
    direction: Literal["ram", "anti-ram", "full"] = "full",
    map_spacing: int = 4,
) -> RectangularSkyMap:
    """
    Project Hi PSET data into a rectangular sky map.

    Parameters
    ----------
    psets : list of str or pathlib.Path
        List of input PSETs to make a map from.
    geometric_factors_path : str or pathlib.Path
        Where to get the geometric factors from.
    esa_energies_path : str or pathlib.Path
        Where to get the energies from.
    cg_corrected : bool, Optional
        Whether to apply Compton-Getting correction to the energies. Defaults to
        False.
    direction : str, Optional
        Apply filtering to PSET data include ram or anti-ram or full spin data.
        Defaults to "full".
    map_spacing : int, Optional
        Pixel spacing, in degrees, of the output map in degrees. Defaults to 4.

    Returns
    -------
    sky_map : RectangularSkyMap
        The sky map with all the PSET data projected into the map.
    """
    rect_map = RectangularSkyMap(
        spacing_deg=map_spacing, spice_frame=SpiceFrame.ECLIPJ2000
    )

    # TODO: Implement Compton-Getting correction
    if cg_corrected:
        raise NotImplementedError
    # TODO: Implement directional filtering
    if direction != "full":
        raise NotImplementedError

    for pset_path in psets:
        logger.info(f"Processing {pset_path}")
        pset = HiPointingSet(pset_path)

        # Background rate and uncertainty are exposure time weighted means in
        # the map.
        for var in VARS_TO_EXPOSURE_TIME_AVERAGE:
            pset.data[var] *= pset.data["exposure_factor"]

        # Project (bin) the PSET variables into the map pixels
        rect_map.project_pset_values_to_map(
            pset,
            ["counts", "exposure_factor", "bg_rates", "bg_rates_unc", "obs_date"],
        )

    # Finish the exposure time weighted mean calculation of backgrounds
    # Allow divide by zero to fill set pixels with zero exposure time to NaN
    with np.errstate(divide="ignore"):
        for var in VARS_TO_EXPOSURE_TIME_AVERAGE:
            rect_map.data_1d[var] /= rect_map.data_1d["exposure_factor"]

    rect_map.data_1d.update(calculate_ena_signal_rates(rect_map.data_1d))
    rect_map.data_1d.update(
        calculate_ena_intensity(
            rect_map.data_1d, geometric_factors_path, esa_energies_path
        )
    )

    rect_map.data_1d["obs_date"].data = rect_map.data_1d["obs_date"].data.astype(
        np.int64
    )
    # TODO: Figure out how to compute obs_date_range (stddev of obs_date)
    rect_map.data_1d["obs_date_range"] = xr.zeros_like(rect_map.data_1d["obs_date"])

    # Rename and convert coordinate from esa_energy_step energy
    # TODO: the correct conversion from esa_energy_step to esa_energy
    esa_energy_step_conversion = (np.arange(10, dtype=float) + 1) * 1000
    rect_map.data_1d = rect_map.data_1d.rename({"esa_energy_step": "energy"})
    rect_map.data_1d = rect_map.data_1d.assign_coords(
        energy=esa_energy_step_conversion[rect_map.data_1d["energy"].values]
    )
    # Set the energy_step_delta values
    # TODO: get the correct energy delta values (they are set to NaN) in
    #    rect_map.build_cdf_dataset()

    rect_map.data_1d = rect_map.data_1d.drop("esa_energy_step_label")

    return rect_map


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
        Where to get the energies from.

    Returns
    -------
    intensity_vars : dict[str, xarray.DataArray]
        ENA Intensity with statistical and systematic uncertainties.
    """
    # TODO: Implement geometric factor lookup
    if geometric_factors_path:
        raise NotImplementedError
    geometric_factor = xr.DataArray(
        np.ones((map_ds["esa_energy_step"].size, map_ds["calibration_prod"].size)),
        coords=[map_ds["esa_energy_step"], map_ds["calibration_prod"]],
    )
    # TODO: Implement esa energies lookup
    if esa_energies_path:
        raise NotImplementedError
    esa_energy = xr.ones_like(map_ds["esa_energy_step"])

    # Convert ENA Signal Rate to Flux
    intensity_vars = {
        "ena_intensity": map_ds["ena_signal_rates"] / (geometric_factor * esa_energy),
        "ena_intensity_stat_unc": map_ds["ena_signal_rate_stat_unc"]
        / geometric_factor
        / esa_energy,
        "ena_intensity_sys_err": map_ds["bg_rates_unc"]
        / (geometric_factor * esa_energy),
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
