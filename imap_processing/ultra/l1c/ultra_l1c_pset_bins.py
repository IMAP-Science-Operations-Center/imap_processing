"""Module to create pointing sets."""

import astropy_healpix.healpy as hp
import numpy as np
import pandas
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy import interpolate
from scipy.interpolate import interp1d

from imap_processing.spice.geometry import (
    SpiceFrame,
    cartesian_to_spherical,
    imap_state,
)
from imap_processing.spice.spin import get_spacecraft_spin_phase, get_spin_angle
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_geometric_factor,
    get_image_params,
)
from imap_processing.ultra.l1b.ultra_l1b_culling import (
    get_pulses_per_spin,
    get_spin_and_duration,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    get_efficiency,
    get_efficiency_interpolator,
)

# TODO: add species binning.
FILLVAL_FLOAT32 = -1.0e31


def build_energy_bins() -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    """
    Build energy bin boundaries.

    Returns
    -------
    intervals : list[tuple[float, float]]
        Energy bins.
    energy_midpoints : np.ndarray
        Array of energy bin midpoints.
    energy_bin_geometric_means : np.ndarray
        Array of geometric means of energy bins.
    """
    # Create energy bins.
    energy_bin_edges = np.array(UltraConstants.CULLING_ENERGY_BIN_EDGES)
    energy_midpoints = (energy_bin_edges[:-1] + energy_bin_edges[1:]) / 2

    intervals = [
        (float(energy_bin_edges[i]), float(energy_bin_edges[i + 1]))
        for i in range(len(energy_bin_edges) - 1)
    ]
    energy_bin_geometric_means = np.sqrt(energy_bin_edges[:-1] * energy_bin_edges[1:])

    return intervals, energy_midpoints, energy_bin_geometric_means


def get_energy_delta_minus_plus() -> tuple[NDArray, NDArray]:
    """
    Calculate the energy_delta_minus and energy_delta_plus for use in the CDF.

    Returns
    -------
    bins_energy_delta_minus : np.ndarray
        Array of energy_delta_minus values.
    bins_energy_delta_plus : np.ndarray
        Array of energy_delta_plus values.

    Notes
    -----
    Calculates as the following:
    energy_delta_minus=abs(bin_geom_mean - bin_lower)
    energy_delta_plus=abs(bin_upper - bin_geom_mean)
    where bin_upper and bin_lower are the upper and lower bounds of the energy bins
    and bin_geom_mean is the geometric mean of the energy bin.
    """
    bins, _, bin_geom_means = build_energy_bins()
    bins_energy_delta_plus, bins_energy_delta_minus = [], []
    for bin_edges, bin_geom_mean in zip(bins, bin_geom_means, strict=False):
        bins_energy_delta_plus.append(bin_edges[1] - bin_geom_mean)
        bins_energy_delta_minus.append(bin_geom_mean - bin_edges[0])
    return abs(np.array(bins_energy_delta_minus)), abs(np.array(bins_energy_delta_plus))


def get_spacecraft_histogram(
    vhat: tuple[np.ndarray, np.ndarray, np.ndarray],
    energy: np.ndarray,
    energy_bin_edges: list[tuple[float, float]],
    nside: int = 128,
    nested: bool = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Compute a 2D histogram of the particle data using HEALPix binning.

    Parameters
    ----------
    vhat : tuple[np.ndarray, np.ndarray, np.ndarray]
        The x,y,z-components of the unit velocity vector.
    energy : np.ndarray
        The particle energy.
    energy_bin_edges : list[tuple[float, float]]
        Array of energy bin edges.
    nside : int, optional
        The nside parameter of the Healpix tessellation.
        Default is 128.
    nested : bool, optional
        Whether the Healpix tessellation is nested. Default is False.

    Returns
    -------
    hist : np.ndarray
        A 2D histogram array with shape (n_pix, n_energy_bins).
    latitude : np.ndarray
        Array of latitude values.
    longitude : np.ndarray
        Array of longitude values.
    n_pix : int
        Number of healpix pixels.

    Notes
    -----
    The histogram will work properly for overlapping energy bins, i.e.
    the same energy value can fall into multiple bins if the intervals overlap.

    azimuthal angle [0, 360], elevation angle [-90, 90]
    """
    # vhat = direction in which particle is traveling
    # Make negative to see where it came from
    spherical_coords = cartesian_to_spherical(-np.array(vhat), degrees=True)
    az, el = (
        spherical_coords[..., 1],
        spherical_coords[..., 2],
    )

    # Compute number of HEALPix pixels that cover the sphere
    n_pix = hp.nside2npix(nside)

    # Calculate the corresponding longitude (az) latitude (el)
    # center coordinates
    longitude, latitude = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)

    # Get HEALPix pixel indices for each event
    # HEALPix expects latitude in [-90, 90] so we don't need to change elevation
    hpix_idx = hp.ang2pix(nside, az, el, nest=nested, lonlat=True)

    # Initialize histogram: (n_energy_bins, n_HEALPix pixels)
    hist = np.zeros((len(energy_bin_edges), n_pix))

    # Bin data in energy & HEALPix space
    for i, (e_min, e_max) in enumerate(energy_bin_edges):
        mask = (energy >= e_min) & (energy < e_max)
        # Only count the events that fall within the energy bin
        hist[i, :] += np.bincount(hpix_idx[mask], minlength=n_pix).astype(np.float64)

    return hist, latitude, longitude, n_pix


def get_spacecraft_count_rate_uncertainty(hist: NDArray, exposure: NDArray) -> NDArray:
    """
    Calculate the count rate uncertainty for HEALPix-binned data.

    Parameters
    ----------
    hist : NDArray
        A 2D histogram array with shape (n_pix, n_energy_bins).
    exposure : NDArray
        A 2D array of exposure times with shape (n_pix, n_energy_bins).

    Returns
    -------
    count_rate_uncertainty : NDArray
        Rate uncertainty with shape (n_pix, n_energy_bins) (counts/sec).

    Notes
    -----
    These calculations were based on Eqn 15 from the IMAP-Ultra Algorithm Document.
    """
    count_uncertainty = np.sqrt(hist)

    rate_uncertainty = np.zeros_like(hist)
    valid = exposure > 0
    rate_uncertainty[valid] = count_uncertainty[valid] / exposure[valid]

    return rate_uncertainty


def get_deadtime_ratios(sectored_rates_ds: xr.Dataset) -> xr.DataArray:
    """
    Compute the dead time ratio at each sector.

    A reduction in exposure time (duty cycle) is caused by the flight hardware listening
    for coincidence events that never occur, due to singles starts predominantly from UV
    radiation. The static exposure time for a given Pointing should be reduced by this
    spatially dependent exposure time reduction factor (the dead time). Further
    description is available in section 3.4.3 of the IMAP-Ultra Algorithm Document.

    Parameters
    ----------
    sectored_rates_ds : xarray.Dataset
        Dataset containing sector mode image rates data.

    Returns
    -------
    dead_time_ratio : xarray.DataArray
        Dead time correction factor for each sector.
    """
    # Compute the correction factor at each sector
    a = sectored_rates_ds.fifo_valid_events / (
        1
        - (sectored_rates_ds.event_active_time + 2 * sectored_rates_ds.start_pos) * 1e-7
    )

    start_full = sectored_rates_ds.start_rf + sectored_rates_ds.start_lf
    b = a * np.exp(start_full * 1e-7 * 5)

    coin_stop_nd = (
        sectored_rates_ds.coin_tn
        + sectored_rates_ds.coin_bn
        - sectored_rates_ds.stop_tn
        - sectored_rates_ds.stop_bn
    )

    corrected_valid_events = b * np.exp(1e-7 * 8 * coin_stop_nd)

    # Compute dead time ratio
    dead_time_ratios = sectored_rates_ds.fifo_valid_events / corrected_valid_events

    return dead_time_ratios


def get_sectored_rates(rates_ds: xr.Dataset, params_ds: xr.Dataset) -> xr.Dataset:
    """
    Filter rates dataset to only include sector mode data.

    Parameters
    ----------
    rates_ds : xarray.Dataset
        Dataset containing image rates data.
    params_ds : xarray.Dataset
        Dataset containing image parameters data.

    Returns
    -------
    rates : xarray.Dataset
        Rates dataset with only the sector mode data.
    """
    # Find indices in which the parameters dataset, indicates that ULTRA was in
    # sector mode. At the normal 15-second spin period, each 24° sector takes ~1 second.

    # This means that data was collected as a function of spin allowing for fine grained
    # rate analysis.
    sector_mode_start_inds = np.where(params_ds["imageratescadence"] == 3)[0]
    # get the sector mode start and stop indices
    sector_mode_stop_inds = sector_mode_start_inds + 1
    # get the sector mode start and stop times
    mode_3_start = params_ds["epoch"].values[sector_mode_start_inds]

    # if the last mode is a sector mode, we can assume that the sector data goes through
    # the end of the dataset, so we append np.inf to the end of the last time range.
    if sector_mode_stop_inds[-1] == len(params_ds["epoch"]):
        mode_3_end = np.append(
            params_ds["epoch"].values[sector_mode_stop_inds[:-1]], np.inf
        )
    else:
        mode_3_end = params_ds["epoch"].values[sector_mode_stop_inds]

    # Build a list of conditions for each sector mode time range
    conditions = [
        (rates_ds["epoch"] >= start) & (rates_ds["epoch"] < end)
        for start, end in zip(mode_3_start, mode_3_end, strict=False)
    ]

    sector_mode_mask = np.logical_or.reduce(conditions)
    return rates_ds.isel(epoch=sector_mode_mask)


def get_deadtime_ratios_by_spin_phase(
    sectored_rates: xr.Dataset,
) -> np.ndarray:
    """
    Calculate nominal deadtime ratios at every spin phase step (1ms res).

    Parameters
    ----------
    sectored_rates : xarray.Dataset
        Dataset containing sector mode image rates data.

    Returns
    -------
    numpy.ndarray
        Nominal deadtime ratios at every spin phase step (1ms res).
    """
    deadtime_ratios = get_deadtime_ratios(sectored_rates)
    # Get the spin phase at the start of each sector rate measurement
    spin_phases = np.asarray(
        get_spin_angle(
            get_spacecraft_spin_phase(np.array(sectored_rates.epoch.data)), degrees=True
        )
    )
    # Assume the sectored rate data is evenly spaced in time, and find the middle spin
    # phase value for each sector.
    # The center spin phase is the closest / most accurate spin phase.
    # There are 24 spin phases per sector so the nominal middle sector spin phases
    # would be: array([ 12., 36., ..., 300., 324.]) for 15 sectors.
    spin_phases_centered = (spin_phases[:-1] + spin_phases[1:]) / 2
    # Assume the last sector is nominal because we dont have enough data to determine
    # the spin phase at the end of the last sector.
    # TODO: is this assumption valid?
    # Add the last spin phase value + half of a nominal sector.
    spin_phases_centered = np.append(spin_phases_centered, spin_phases[-1] + 12)
    # Wrap any spin phases > 360 back to [0, 360]
    spin_phases_centered = spin_phases_centered % 360
    # Create a dataset with spin phases and dead time ratios
    deadtime_by_spin_phase = xr.Dataset(
        {"deadtime_ratio": deadtime_ratios},
        coords={
            "spin_phase": xr.DataArray(np.array(spin_phases_centered), dims="epoch")
        },
    )

    # Sort the dataset by spin phase (ascending order)
    deadtime_by_spin_phase = deadtime_by_spin_phase.sortby("spin_phase")
    # Group by spin phase and calculate the median dead time ratio for each phase
    deadtime_medians = deadtime_by_spin_phase.groupby("spin_phase").median(skipna=True)

    if np.any(np.isnan(deadtime_medians["deadtime_ratio"].values)):
        raise ValueError(
            "Dead time ratios contain NaN values, cannot create interpolator."
        )
    interpolator = interpolate.PchipInterpolator(
        deadtime_medians["spin_phase"].values, deadtime_medians["deadtime_ratio"].values
    )
    # Calculate the nominal spin phases at 1 ms resolution and query the pchip
    # interpolator to get the deadtime ratios.
    steps = 15 * 1000  # 15 seconds at 1 ms resolution
    nominal_spin_phases_1ms_res = np.arange(0, 360, 360 / steps)
    return interpolator(nominal_spin_phases_1ms_res)


def apply_deadtime_correction(
    exposure_pointing: pandas.DataFrame,
    deadtime_ratios: np.ndarray,
    pixels_below_scattering: list,
) -> np.ndarray:
    """
    Adjust the exposure time at each pixel to account for dead time.

    Parameters
    ----------
    exposure_pointing : pandas.DataFrame
        Exposure data.
    deadtime_ratios : PchipInterpolator
        Interpolating function for dead time ratios.
    pixels_below_scattering : list
        A Nested list of arrays indicating pixels within the scattering threshold.
        The outer list indicates spin phase steps, the middle list indicates energy
        bins, and the inner arrays contain indices indicating pixels that are below
        the FWHM scattering threshold.

    Returns
    -------
    exposure_pointing_adjusted : np.ndarray
        Adjusted exposure times accounting for dead time.
    """
    # Get energy bin geometric means
    energy_bin_geometric_means = build_energy_bins()[2]
    # Exposure time should now be of shape (npix, energy)
    exposure_pointing = np.repeat(
        exposure_pointing.to_numpy()[np.newaxis, :],
        len(energy_bin_geometric_means),
        axis=0,
    )
    # nominal spin phase step.
    nominal_ms_step = 15 / len(pixels_below_scattering)  # time step
    # Query the dead-time ratio and apply the nominal exposure time to pixels in the FOR
    # and below the scattering threshold
    # Loop through the spin phase steps. This is spinning the spacecraft by nominal
    # 1 ms steps in the despun frame.
    for i, pixels_at_spin in enumerate(pixels_below_scattering):
        # Loop through energy bins
        for energy_bin_idx in range(len(energy_bin_geometric_means)):
            pixels_at_energy_and_spin = pixels_at_spin[energy_bin_idx]
            if pixels_at_energy_and_spin.size == 0:
                continue
            # Apply the nominal exposure time (1 ms) scaled by the deadtime ratio to
            # every pixel in the FOR, that is below the FWHM scattering threshold,
            exposure_pointing[energy_bin_idx, pixels_at_energy_and_spin] += (
                nominal_ms_step * deadtime_ratios[i]
            )

    return exposure_pointing


def get_spacecraft_exposure_times(
    constant_exposure: pandas.DataFrame,
    rates_dataset: xr.Dataset,
    params_dataset: xr.Dataset,
    pixels_below_scattering: list[list],
) -> NDArray:
    """
    Compute exposure times for HEALPix pixels.

    Parameters
    ----------
    constant_exposure : pandas.DataFrame
        Exposure data.
    rates_dataset : xarray.Dataset
        Dataset containing image rates data.
    params_dataset : xarray.Dataset
        Dataset containing image parameters data.
    pixels_below_scattering : list
        List of lists indicating pixels within the scattering threshold.
        The outer list indicates spin phase steps, the middle list indicates energy
        bins, and the inner list contains pixel indices indicating pixels that are
        below the FWHM scattering threshold.

    Returns
    -------
    exposure_pointing : np.ndarray
        Total exposure times of pixels in a
        Healpix tessellation of the sky
        in the pointing (dps) frame.
    """
    # TODO: use the universal spin table and
    #  universal pointing table here to determine actual number of spins
    sectored_rates = get_sectored_rates(rates_dataset, params_dataset)
    nominal_deadtime_ratios = get_deadtime_ratios_by_spin_phase(sectored_rates)
    # TODO save nominal_deadtime_ratios to the pset.
    exposure_pointing = (
        constant_exposure["Exposure Time"] * 5760
    )  # 5760 spins per pointing (for now)
    exposure_pointing_adjusted = apply_deadtime_correction(
        exposure_pointing, nominal_deadtime_ratios, pixels_below_scattering
    )
    return exposure_pointing_adjusted


def get_efficiencies_and_geometric_function(
    pixels_below_scattering: list[list],
    theta_and_phi: np.ndarray,
    ancillary_files: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the geometric factor and efficiency for each pixel and energy bin.

    The results are averaged over all spin phases.

    Parameters
    ----------
    pixels_below_scattering : list
        List of lists indicating pixels within the scattering threshold.
        The outer list indicates spin phase steps, the middle list indicates energy
        bins, and the inner list contains pixel indices indicating pixels that are
        below the FWHM scattering threshold.
    theta_and_phi : np.ndarray
        Array of theta and phi values for each pixel. First column is theta,
        second is phi.
    ancillary_files : dict
        Dictionary containing ancillary files.

    Returns
    -------
    gf_summation : np.ndarray
        Summation of geometric factors for each pixel and energy bin.
    eff_summation : np.ndarray
        Summation of efficiencies for each pixel and energy bin.
    """
    # Load callable efficiency interpolator function
    eff_interpolator = get_efficiency_interpolator(ancillary_files)
    # Get energy bin geometric means
    energy_bin_geometric_means = build_energy_bins()[2]
    energy_bins = len(energy_bin_geometric_means)
    # Number of pixels is the length of theta or phi
    npix = theta_and_phi.shape[0]
    # Initialize summation arrays for geometric factors and efficiencies
    gf_summation = np.zeros((energy_bins, npix))
    eff_summation = np.zeros((energy_bins, npix))
    sample_count = np.zeros((energy_bins, npix))
    # Compute gf and eff for these theta/phi pairs
    gf_values = get_geometric_factor(
        ancillary_files,
        "l1b-sensor-gf-blades",
        theta_and_phi[:, 0],  # Theta
        theta_and_phi[:, 1],  # Phi
        np.zeros(theta_and_phi.shape[0]).astype(np.uint16),
    )
    # Loop through spin phases
    for pixels_at_spin in pixels_below_scattering:
        # Loop through energy bins
        for energy_bin_idx in range(energy_bins):
            pixel_inds = pixels_at_spin[energy_bin_idx]
            if pixel_inds.size == 0:
                continue
            energy = energy_bin_geometric_means[energy_bin_idx]
            # Get theta and phi vals for pixels in the FOR at this spin phase
            theta_vals = theta_and_phi[pixel_inds, 0]
            phi_vals = theta_and_phi[pixel_inds, 1]
            eff_values = get_efficiency(
                np.full(phi_vals.shape, energy),
                phi_vals,
                theta_vals,
                ancillary_files,
                interpolator=eff_interpolator,
            )

            # Accumulate
            gf_summation[energy_bin_idx, pixel_inds] += gf_values[pixel_inds]
            eff_summation[energy_bin_idx, pixel_inds] += eff_values
            sample_count[energy_bin_idx, pixel_inds] += 1

    # return averaged geometric factors and efficiencies across all spin phases
    # These are now energy dependent.
    non_zero = sample_count > 0
    gf_averaged = gf_summation[non_zero] / sample_count[non_zero]
    eff_averaged = eff_summation[non_zero] / sample_count[non_zero]
    return gf_averaged, eff_averaged


def get_helio_exposure_times(
    time: np.ndarray,
    df_exposure: pd.DataFrame,
    nside: int = 128,
    nested: bool = False,
) -> NDArray:
    """
    Compute a 2D (Healpix index, energy) array of exposure in the helio frame.

    Parameters
    ----------
    time : np.ndarray
        Median time of pointing in et.
    df_exposure : pd.DataFrame
        Spacecraft exposure in healpix coordinates.
    nside : int, optional
        The nside parameter of the Healpix tessellation (default is 128).
    nested : bool, optional
        Whether the Healpix tessellation is nested (default is False).

    Returns
    -------
    helio_exposure : np.ndarray
        A 2D array of shape (npix, n_energy_bins).

    Notes
    -----
    These calculations are performed once per pointing.
    """
    # Get energy midpoints.
    _, energy_midpoints, _ = build_energy_bins()
    # Extract (RA/Dec) and exposure from the spacecraft frame.
    ra = df_exposure["Right Ascension (deg)"].values
    dec = df_exposure["Declination (deg)"].values
    exposure_flat = df_exposure["Exposure Time"].values

    # The Cartesian state vector representing the position and velocity of the
    # IMAP spacecraft.
    state = imap_state(time, ref_frame=SpiceFrame.IMAP_DPS)

    # Extract the velocity part of the state vector
    spacecraft_velocity = state[3:6]
    # Convert (RA, Dec) angles into 3D unit vectors.
    # Each unit vector represents a direction in the sky where the spacecraft observed
    # and accumulated exposure time.
    unit_dirs = hp.ang2vec(ra, dec, lonlat=True).T  # Shape (N, 3)

    # Initialize output array.
    # Each row corresponds to a HEALPix pixel, and each column to an energy bin.
    npix = hp.nside2npix(nside)
    helio_exposure = np.zeros((len(energy_midpoints), npix))

    # Loop through energy bins and compute transformed exposure.
    for i, energy_midpoint in enumerate(energy_midpoints):
        # Convert the midpoint energy to a velocity (km/s).
        # Based on kinetic energy equation: E = 1/2 * m * v^2.
        energy_velocity = (
            np.sqrt(2 * energy_midpoint * UltraConstants.KEV_J / UltraConstants.MASS_H)
            / 1e3
        )

        # Use Galilean Transform to transform the velocity wrt spacecraft
        # to the velocity wrt heliosphere.
        # energy_velocity * cartesian -> apply the magnitude of the velocity
        # to every position on the grid in the despun grid.
        helio_velocity = spacecraft_velocity.reshape(1, 3) + energy_velocity * unit_dirs

        # Normalized vectors representing the direction of the heliocentric velocity.
        helio_normalized = helio_velocity / np.linalg.norm(
            helio_velocity, axis=1, keepdims=True
        )

        # Convert Cartesian heliocentric vectors into spherical coordinates.
        # Result: azimuth (longitude) and elevation (latitude) in degrees.
        helio_spherical = cartesian_to_spherical(helio_normalized)
        az, el = helio_spherical[:, 1], helio_spherical[:, 2]

        # Convert azimuth/elevation directions to HEALPix pixel indices.
        hpix_idx = hp.ang2pix(nside, az, el, nest=nested, lonlat=True)

        # Accumulate exposure values into HEALPix pixels for this energy bin.
        helio_exposure[i, :] = np.bincount(
            hpix_idx, weights=exposure_flat, minlength=npix
        )

    return helio_exposure


def get_spacecraft_sensitivity(
    efficiencies: pandas.DataFrame,
    geometric_function: pandas.DataFrame,
) -> tuple[pandas.DataFrame, NDArray, NDArray, NDArray]:
    """
    Compute sensitivity as efficiency * geometric factor.

    Parameters
    ----------
    efficiencies : pandas.DataFrame
        Efficiencies at different energy levels.
    geometric_function : pandas.DataFrame
        Geometric function.

    Returns
    -------
    pointing_sensitivity : pandas.DataFrame
        Sensitivity with dimensions (HEALPIX pixel_number, energy).
    energy_vals : NDArray
        Energy values of dataframe.
    right_ascension : NDArray
        Right ascension (longitude/azimuth) values of dataframe (0 - 360 degrees).
    declination : NDArray
        Declination (latitude/elevation) values of dataframe (-90 to 90 degrees).
    """
    # Exclude "Right Ascension (deg)" and "Declination (deg)" from the multiplication
    energy_columns = [
        col
        for col in efficiencies.columns
        if col not in ["Right Ascension (deg)", "Declination (deg)"]
    ]
    sensitivity = efficiencies[energy_columns].mul(
        geometric_function["Response (cm2-sr)"].values, axis=0
    )

    right_ascension = efficiencies["Right Ascension (deg)"]
    declination = efficiencies["Declination (deg)"]

    energy_vals = np.array([float(col.replace("keV", "")) for col in energy_columns])

    return sensitivity, energy_vals, right_ascension, declination


def grid_sensitivity(
    efficiencies: pandas.DataFrame,
    geometric_function: pandas.DataFrame,
    energy: float,
) -> NDArray:
    """
    Grid the sensitivity.

    Parameters
    ----------
    efficiencies : pandas.DataFrame
        Efficiencies at different energy levels.
    geometric_function : pandas.DataFrame
        Geometric function.
    energy : float
        Energy to which we are interpolating.

    Returns
    -------
    interpolated_sensitivity : np.ndarray
        Sensitivity with dimensions (HEALPIX pixel_number, 1).
    """
    sensitivity, energy_vals, right_ascension, declination = get_spacecraft_sensitivity(
        efficiencies, geometric_function
    )

    # Create interpolator over energy dimension for each pixel (axis=1)
    interp_func = interp1d(
        energy_vals,
        sensitivity.values,
        axis=1,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Interpolate to energy
    interpolated = interp_func(energy)
    interpolated = np.where(np.isnan(interpolated), FILLVAL_FLOAT32, interpolated)

    return interpolated


def interpolate_sensitivity(
    efficiencies: pd.DataFrame,
    geometric_function: pd.DataFrame,
    nside: int = 128,
) -> NDArray:
    """
    Interpolate the sensitivity and bin it in HEALPix space.

    Parameters
    ----------
    efficiencies : pandas.DataFrame
        Efficiencies at different energy levels.
    geometric_function : pandas.DataFrame
        Geometric function.
    nside : int, optional
        Healpix nside resolution (default is 128).

    Returns
    -------
    interpolated_sensitivity : np.ndarray
        Array of shape (n_energy_bins, n_healpix_pixels).
    """
    _, _, energy_bin_geometric_means = build_energy_bins()
    npix = hp.nside2npix(nside)

    interpolated_sensitivity = np.full(
        (len(energy_bin_geometric_means), npix), FILLVAL_FLOAT32
    )

    for i, energy in enumerate(energy_bin_geometric_means):
        pixel_sensitivity = grid_sensitivity(
            efficiencies, geometric_function, energy
        ).flatten()
        interpolated_sensitivity[i, :] = pixel_sensitivity

    return interpolated_sensitivity


def get_helio_sensitivity(
    time: np.ndarray,
    efficiencies: pandas.DataFrame,
    geometric_function: pandas.DataFrame,
    nside: int = 128,
    nested: bool = False,
) -> NDArray:
    """
    Compute a 2D (Healpix index, energy) array of sensitivity in the helio frame.

    Parameters
    ----------
    time : np.ndarray
        Median time of pointing in et.
    efficiencies : pandas.DataFrame
        Efficiencies at different energy levels.
    geometric_function : pandas.DataFrame
        Geometric function.
    nside : int, optional
        The nside parameter of the Healpix tessellation (default is 128).
    nested : bool, optional
        Whether the Healpix tessellation is nested (default is False).

    Returns
    -------
    helio_sensitivity : np.ndarray
        A 2D array of shape (npix, n_energy_bins).

    Notes
    -----
    These calculations are performed once per pointing.
    """
    # Get energy midpoints.
    _, energy_midpoints, _ = build_energy_bins()

    # Get sensitivity on the spacecraft grid
    _, _, ra, dec = get_spacecraft_sensitivity(efficiencies, geometric_function)

    # The Cartesian state vector representing the position and velocity of the
    # IMAP spacecraft.
    state = imap_state(time, ref_frame=SpiceFrame.IMAP_DPS)

    # Extract the velocity part of the state vector
    spacecraft_velocity = state[3:6]
    # Convert (RA, Dec) angles into 3D unit vectors.
    # Each unit vector represents a direction in the sky where the spacecraft observed
    # and accumulated sensitivity.
    unit_dirs = hp.ang2vec(ra, dec, lonlat=True).T  # Shape (N, 3)

    # Initialize output array.
    # Each row corresponds to a HEALPix pixel, and each column to an energy bin.
    npix = hp.nside2npix(nside)
    helio_sensitivity = np.zeros((len(energy_midpoints), npix))

    # Loop through energy bins and compute transformed sensitivity.
    for i, energy in enumerate(energy_midpoints):
        # Convert the midpoint energy to a velocity (km/s).
        # Based on kinetic energy equation: E = 1/2 * m * v^2.
        energy_velocity = (
            np.sqrt(2 * energy * UltraConstants.KEV_J / UltraConstants.MASS_H) / 1e3
        )

        # Use Galilean Transform to transform the velocity wrt spacecraft
        # to the velocity wrt heliosphere.
        # energy_velocity * cartesian -> apply the magnitude of the velocity
        # to every position on the grid in the despun grid.
        helio_velocity = spacecraft_velocity.reshape(1, 3) + energy_velocity * unit_dirs

        # Normalized vectors representing the direction of the heliocentric velocity.
        helio_normalized = helio_velocity / np.linalg.norm(
            helio_velocity, axis=1, keepdims=True
        )

        # Convert Cartesian heliocentric vectors into spherical coordinates.
        # Result: azimuth (longitude) and elevation (latitude) in degrees.
        helio_spherical = cartesian_to_spherical(helio_normalized)
        az, el = helio_spherical[:, 1], helio_spherical[:, 2]

        # Convert azimuth/elevation directions to HEALPix pixel indices.
        hpix_idx = hp.ang2pix(nside, az, el, nest=nested, lonlat=True)
        gridded_sensitivity = grid_sensitivity(efficiencies, geometric_function, energy)

        # Accumulate sensitivity values into HEALPix pixels for this energy bin.
        helio_sensitivity[i, :] = np.bincount(
            hpix_idx, weights=gridded_sensitivity, minlength=npix
        )

    return helio_sensitivity


def get_spacecraft_background_rates(
    rates_dataset: xr.Dataset,
    sensor: str,
    ancillary_files: dict,
    energy_bin_edges: list[tuple[float, float]],
    cullingmask_spin_number: NDArray,
    nside: int = 128,
) -> NDArray:
    """
    Calculate background rates based on the provided parameters.

    Parameters
    ----------
    rates_dataset : xr.Dataset
        Rates dataset.
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.
    energy_bin_edges : list[tuple[float, float]]
        Energy bin edges.
    cullingmask_spin_number : NDArray
        Goodtime spins.
        Ex. imap_ultra_l1b_45sensor-cullingmask[0]["spin_number"]
        This is used to determine the number of pulses per spin.
    nside : int, optional
        The nside parameter of the Healpix tessellation (default is 128).

    Returns
    -------
    background_rates : NDArray of shape (n_energy_bins, n_HEALPix pixels)
        Calculated background rates.

    Notes
    -----
    See Eqn. 3, 8, and 20 in the Algorithm Document for the equation.
    """
    pulses = get_pulses_per_spin(rates_dataset)
    # Pulses for the pointing.
    etof_min = get_image_params("eTOFMin", sensor, ancillary_files)
    etof_max = get_image_params("eTOFMax", sensor, ancillary_files)
    spin_number, _ = get_spin_and_duration(
        rates_dataset["shcoarse"], rates_dataset["spin"]
    )

    # Get dmin for PH (mm).
    dmin_ctof = UltraConstants.DMIN_PH_CTOF

    # Compute number of HEALPix pixels that cover the sphere
    n_pix = hp.nside2npix(nside)

    # Initialize background rate array: (n_energy_bins, n_HEALPix pixels)
    background_rates = np.zeros((len(energy_bin_edges), n_pix))

    # Only select pulses from goodtimes.
    goodtime_mask = np.isin(spin_number, cullingmask_spin_number)
    mean_start_pulses = np.mean(pulses.start_pulses[goodtime_mask])
    mean_stop_pulses = np.mean(pulses.stop_pulses[goodtime_mask])
    mean_coin_pulses = np.mean(pulses.coin_pulses[goodtime_mask])

    for i, (e_min, e_max) in enumerate(energy_bin_edges):
        # Calculate ctof for the energy bin boundaries by combining Eqn. 3 and 8.
        # Compute speed for min and max energy using E = 1/2mv^2 -> v = sqrt(2E/m)
        vmin = np.sqrt(2 * e_min * UltraConstants.KEV_J / UltraConstants.MASS_H)  # m/s
        vmax = np.sqrt(2 * e_max * UltraConstants.KEV_J / UltraConstants.MASS_H)  # m/s
        # Compute cTOF = dmin / v
        # Multiply times 1e-3 to convert to m.
        ctof_min = dmin_ctof * 1e-3 / vmax * 1e-9  # Convert to ns
        ctof_max = dmin_ctof * 1e-3 / vmin * 1e-9  # Convert to ns

        background_rates[i, :] = (
            np.abs(ctof_max - ctof_min)
            * (etof_max - etof_min)
            * mean_start_pulses
            * mean_stop_pulses
            * mean_coin_pulses
        ) / 30.0

    return background_rates
