"""Module to create pointing sets."""

import astropy_healpix.healpy as hp
import numpy as np
import pandas
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from imap_processing.spice.geometry import (
    SpiceFrame,
    cartesian_to_spherical,
    imap_state,
)
from imap_processing.ultra.constants import UltraConstants

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
    for bin_edges, bin_geom_mean in zip(bins, bin_geom_means):
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


def get_helio_histogram(
    time: NDArray,
    vhat: NDArray,
    energy: NDArray,
    energy_bin_edges: list[tuple[float, float]],
    nside: int = 128,
    nested: bool = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Compute a 3D histogram of the particle data using HEALPix binning.

    Parameters
    ----------
    time : np.ndarray
        Median time of pointing in et.
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
    # Compute number of HEALPix pixels that cover the sphere
    n_pix = hp.nside2npix(nside)

    # Calculate the corresponding longitude (az) latitude (el)
    # center coordinates
    longitude, latitude = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)

    # The Cartesian state vector representing the position and velocity of the
    # IMAP spacecraft.
    state = imap_state(time, ref_frame=SpiceFrame.IMAP_DPS)

    # Extract the velocity part of the state vector
    spacecraft_velocity = state[3:6]

    # Initialize histogram: (n_energy_bins, n_HEALPix pixels)
    hist = np.zeros((len(energy_bin_edges), n_pix))

    # Bin data in energy & HEALPix space
    for i, (e_min, e_max) in enumerate(energy_bin_edges):
        # Convert the midpoint energy to a velocity (km/s).
        # Based on kinetic energy equation: E = 1/2 * m * v^2.
        energy_midpoint = (e_min + e_max) / 2
        energy_velocity = (
            np.sqrt(2 * energy_midpoint * UltraConstants.KEV_J / UltraConstants.MASS_H)
            / 1e3
        )

        # Use Galilean Transform to transform the velocity wrt spacecraft
        # to the velocity wrt heliosphere.
        # energy_velocity * cartesian -> apply the magnitude of the velocity
        # to every position on the grid in the despun grid.
        mask = (energy >= e_min) & (energy < e_max)
        vx, vy, vz = vhat.T

        # Select only the particles that fall within the energy bin.
        vx_bin, vy_bin, vz_bin = vx[mask], vy[mask], vz[mask]
        vhat_bin = np.stack((vx_bin, vy_bin, vz_bin), axis=1)
        helio_velocity = spacecraft_velocity.reshape(1, 3) + energy_velocity * vhat_bin

        # Normalized vectors representing the direction of the heliocentric velocity.
        helio_normalized = -helio_velocity / np.linalg.norm(
            helio_velocity, axis=1, keepdims=True
        )

        # Convert Cartesian heliocentric vectors into spherical coordinates.
        # Result: azimuth (longitude) and elevation (latitude) in degrees.
        helio_spherical = cartesian_to_spherical(np.squeeze(helio_normalized))
        helio_spherical = np.atleast_2d(helio_spherical)
        az, el = helio_spherical[:, 1], helio_spherical[:, 2]

        # Convert azimuth/elevation directions to HEALPix pixel indices.
        hpix_idx = hp.ang2pix(nside, az, el, nest=nested, lonlat=True)

        # Only count the events that fall within the energy bin
        hist[i, :] += np.bincount(hpix_idx, minlength=n_pix).astype(np.float64)

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


def get_spacecraft_exposure_times(constant_exposure: pandas.DataFrame) -> NDArray:
    """
    Compute exposure times for HEALPix pixels.

    Parameters
    ----------
    constant_exposure : pandas.DataFrame
        Exposure data.

    Returns
    -------
    exposure_pointing : np.ndarray
        Total exposure times of pixels in a
        Healpix tessellation of the sky
        in the pointing (dps) frame.
    """
    # TODO: use the universal spin table and
    #  universal pointing table here to determine actual number of spins
    exposure_pointing = (
        constant_exposure["Exposure Time"] * 5760
    )  # 5760 spins per pointing (for now)

    return exposure_pointing


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
    helio_exposure = np.zeros((npix, len(energy_midpoints)))

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
        helio_exposure[:, i] = np.bincount(
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
    helio_sensitivity = np.zeros((npix, len(energy_midpoints)))

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
        helio_sensitivity[:, i] = np.bincount(
            hpix_idx, weights=gridded_sensitivity, minlength=npix
        )

    return helio_sensitivity
