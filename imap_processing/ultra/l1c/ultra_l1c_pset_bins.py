"""Module to create pointing sets."""

import astropy_healpix.healpy as hp
import numpy as np
import pandas
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy import interpolate
from scipy.interpolate import PchipInterpolator, interp1d

from imap_processing.spice.geometry import (
    SpiceFrame,
    cartesian_to_spherical,
    frame_transform,
    imap_state,
)
from imap_processing.spice.spin import get_spacecraft_spin_phase, get_spin_angle
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import is_inside_fov

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
    # Calculate energy step
    energy_step = (1 + UltraConstants.ALPHA / 2) / (1 - UltraConstants.ALPHA / 2)

    # Create energy bins.
    energy_bin_edges = UltraConstants.ENERGY_START * energy_step ** np.arange(
        UltraConstants.N_BINS + 1
    )
    # Add a zero to the left side for outliers and round to nearest 3 decimal places.
    energy_bin_edges = np.around(np.insert(energy_bin_edges, 0, 0), 3)
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
    Compute a 3D histogram of the particle data using HEALPix binning.

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
        A 3D histogram array with shape (n_pix, n_energy_bins).
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


def get_spacecraft_background_rates(
    nside: int = 128,
) -> NDArray:
    """
    Calculate background rates.

    Parameters
    ----------
    nside : int, optional
        The nside parameter of the Healpix tessellation (default is 128).

    Returns
    -------
    background_rates : np.ndarray
        Array of background rates.

    Notes
    -----
    This is a placeholder.
    TODO: background rates to be provided by IT.
    """
    npix = hp.nside2npix(nside)
    _, energy_midpoints, _ = build_energy_bins()
    background = np.zeros((len(energy_midpoints), npix))
    return background


def get_helio_background_rates(
    nside: int = 128,
) -> NDArray:
    """
    Calculate background rates.

    Parameters
    ----------
    nside : int, optional
        The nside parameter of the Healpix tessellation (default is 128).

    Returns
    -------
    background_rates : np.ndarray
        Array of background rates.

    Notes
    -----
    This is a placeholder.
    TODO: background rates to be provided by IT.
    """
    npix = hp.nside2npix(nside)
    _, energy_midpoints, _ = build_energy_bins()
    background = np.zeros((len(energy_midpoints), npix))
    return background


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


def get_deadtime_interpolator(
    deadtime_ratios: xr.DataArray, timestamps: xr.DataArray
) -> PchipInterpolator:
    """
    Create PCHIP function for dead time ratio vs spin phase.

    Parameters
    ----------
    deadtime_ratios : xarray.DataArray
        Dead time ratios for each sector.
    timestamps : xarray.DataArray
        Epoch values corresponding to the dead time ratios.

    Returns
    -------
    scipy.interpolate.PchipInterpolator
        Interpolating function for dead time ratios.
    """
    # Get the spin phase at the start of each sector rate measurement
    spin_phases = np.asarray(
        get_spin_angle(get_spacecraft_spin_phase(np.array(timestamps)), degrees=True)
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
    # Return a PCHIP interpolator for the dead time ratios
    return interpolate.PchipInterpolator(
        deadtime_medians["spin_phase"].values, deadtime_medians["deadtime_ratio"].values
    )


def compute_instrument_frame_az_and_el(vecs: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute azimuth and elevation angles in the instrument frame.

    Parameters
    ----------
    vecs : NDArray
        Unit vectors in the instrument frame, shape (N, 3) or (N, boundary_vecs, 3).

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        Theta and phi angles in the instrument frame.
    """
    # convert to theta, phi
    d = 1  # TODO calculate correct d
    r = np.sqrt(vecs[..., 0] ** 2 + vecs[..., 1] ** 2 + d**2)

    # Calculate phi
    phi = np.arctan(vecs[..., 2] / d)
    # Calculate theta
    theta = np.arcsin(vecs[..., 0] / r)
    return theta, phi


def compute_boundary_scale_factor(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute the scale factors using the boundary and center pixel vectors.

    This weighting attempts to approximate how much of the FOV is in a pixel and will
    be used to correct the exposure time.

    Parameters
    ----------
    theta : np.ndarray
        Azimuth angles of the boundary vectors in the instrument frame.
        Shape = (n, boundary_pix + 1). Plus one for the center pixel.
    phi : np.ndarray
        Elevation angles of the boundary vectors in the instrument frame.
        Shape = (n, boundary_pix + 1) Plus one for the center pixel.

    Returns
    -------
    numpy.array
        Scale factor for each pixel, shape = (n,).
    """
    # Get boundary vectors + center pixel that are inside the FOV
    inside_fov = is_inside_fov(theta, phi)  # Shape (n, boundary_pix + 1)
    # For each pixel, return the number that are inside the FOV over
    # the total number of vectors.
    return np.count_nonzero(inside_fov, axis=-1) / theta.shape[-1]


def apply_deadtime_correction(
    exposure_pointing: np.ndarray,
    deadtime_interpolator: PchipInterpolator,
    instrument_id: int,
    nside: int = 128,
    nested: bool = False,
    boundary_pix: int = 8,
) -> np.ndarray:
    """
    Adjust the exposure time at each pixel to account for dead time.

    Parameters
    ----------
    exposure_pointing : np.ndarray
        Total exposure times of pixels in a Healpix tessellation of the sky in the
        pointing (dps) frame.
    deadtime_interpolator : PchipInterpolator
        Interpolating function for dead time ratios.
    instrument_id : int,
        Instrument ID, either 45 or 90.
    nside : int, optional
        HEALPix NSIDE resolution. Default is 128.
    nested : bool, optional
        Whether to use NESTED indexing.
    boundary_pix : int, optional
        Number of boundary pixels to consider for each HEALPix pixel.

    Returns
    -------
    exposure_pointing_adjusted : np.ndarray
        Adjusted exposure times accounting for dead time.
    """
    # Get the correct instrument frame
    instrument_frame = (
        SpiceFrame.IMAP_ULTRA_45 if instrument_id == 45 else SpiceFrame.IMAP_ULTRA_90
    )
    # Get energy bin geometric means
    energy_bin_geometric_means = build_energy_bins()[2]
    # Get pixel unit vectors pointing from the center of the
    # HEALPix sphere to the center of each pixel on the sky.
    npix = hp.nside2npix(nside)
    pixel_vecs = np.column_stack(
        hp.pix2vec(nside, np.arange(npix), nest=nested)
    )  # shape: (npix, 3)

    # boundary_step = boundary_pix / 4
    # boundary_vecs = np.stack(
    #     [
    #         hp.boundaries(nside, px, nest=nested, step=boundary_step).T
    #         for px in range(npix)
    #     ]
    # )  # shape: (npix, boundary_pix, 3)

    # nominal amount the spacecraft would have spun in 1 ms
    nominal_ms_spin = 360 / (15 * 1000)

    # Starting at Spin-Phase = 0, and incrementing in fine steps (1 ms), spin the
    # spacecraft in the despun frame. At each iteration, query the dead-time ratio
    # from the function previously built and apply the nominal exposure time
    et = 798052670.0  # TODO get spin phase 0 time
    for spin_phase in np.arange(0, 360, nominal_ms_spin):
        # Get the vectors in the instrument frame
        pixel_vecs_inst = frame_transform(
            et=et,
            position=pixel_vecs,
            from_frame=SpiceFrame.ECLIPJ2000,
            to_frame=instrument_frame,
        )
        # boundary_vecs_inst = frame_transform(
        #     et=et,
        #     position=boundary_vecs,
        #     from_frame=SpiceFrame.ECLIPJ2000,
        #     to_frame=instrument_frame,
        # )
        # Get Theta/Phi in the instrument frame
        theta, phi = compute_instrument_frame_az_and_el(pixel_vecs_inst)
        # Get mask for pixels in the FOR
        inside_fov = is_inside_fov(phi, theta)
        # theta_in_fov = theta[inside_fov]
        # phi_in_fov = phi[inside_fov]
        # boundary_theta, boundary_phi = compute_instrument_frame_az_and_el(
        #     boundary_vecs_inst[inside_fov]
        # )
        # boundary_scale_factor = compute_boundary_scale_factor(
        #     np.hstack((boundary_theta, theta_in_fov[:, np.newaxis])),
        #     np.hstack((boundary_phi, phi_in_fov[:, np.newaxis])),
        # )
        if any(inside_fov):
            for energy in energy_bin_geometric_means:
                # TODO compute scattering FWHM_Phi & FWHM_Theta

                # fwhm_phi = get
                # fwhm_theta = theta
                # If either Phi FWHM or Theta FWHM > the scattering requirements do not
                # include
                # the instrument frame pixel at the current spin phase
                # TODO uncomment and replace thresholds
                # if fwhm_phi > 10 or fwhm_theta > 10:
                #     continue
                deadtime_ratio = deadtime_interpolator(spin_phase) * energy

                # Apply the nominal exposure time (1 ms) to every pixel in the FOR,
                # scaled by the deadtime ratio
                exposure_pointing[inside_fov] += nominal_ms_spin * deadtime_ratio
                # print("Spin Phase: ", spin_phase, "Deadtime Ratio: ", deadtime_ratio,)
                # increment time by 1ms
                et = et + 0.001

    return exposure_pointing


def get_spacecraft_exposure_times(
    constant_exposure: pandas.DataFrame,
    rates_dataset: xr.Dataset,
    params_dataset: xr.Dataset,
    instrument_id: int,
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
    instrument_id : int
        Instrument ID, either 45 or 90.

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
    deadtime_ratios = get_deadtime_ratios(sectored_rates)
    deadtime_interpolator = get_deadtime_interpolator(
        deadtime_ratios, sectored_rates.epoch.data
    )
    exposure_pointing = (
        constant_exposure["Exposure Time"] * 5760
    )  # 5760 spins per pointing (for now)

    exposure_pointing_adjusted = apply_deadtime_correction(
        exposure_pointing, deadtime_interpolator, instrument_id
    )
    return exposure_pointing_adjusted


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
