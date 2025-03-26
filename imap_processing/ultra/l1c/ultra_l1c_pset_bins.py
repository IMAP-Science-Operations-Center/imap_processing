"""Module to create pointing sets."""

import astropy_healpix.healpy as hp
import numpy as np
import pandas
from numpy.typing import NDArray

from imap_processing.ena_maps.utils.spatial_utils import build_spatial_bins
from imap_processing.spice.geometry import (
    SpiceFrame,
    cartesian_to_spherical,
    imap_state,
    spherical_to_cartesian,
)
from imap_processing.ultra.constants import UltraConstants

# TODO: add species binning.


def build_energy_bins() -> tuple[list[tuple[float, float]], np.ndarray]:
    """
    Build energy bin boundaries.

    Returns
    -------
    intervals : list[tuple[float, float]]
        Energy bins.
    energy_midpoints : np.ndarray
        Array of energy bin midpoints.
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

    return intervals, energy_midpoints


def get_spacecraft_histogram(
    vhat: tuple[np.ndarray, np.ndarray, np.ndarray],
    energy: np.ndarray,
    energy_bin_edges: list[tuple[float, float]],
    nside: int = 128,
    nested: bool = False,
) -> NDArray:
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
        Default is 32.
    nested : bool, optional
        Whether the Healpix tessellation is nested. Default is False.

    Returns
    -------
    hist : np.ndarray
        A 3D histogram array with shape (n_pix, n_energy_bins).

    Notes
    -----
    The histogram will work properly for overlapping energy bins, i.e.
    the same energy value can fall into multiple bins if the intervals overlap.

    azimuthal angle [0, 360], elevation angle [-90, 90]
    """
    spherical_coords = cartesian_to_spherical(vhat, degrees=True)
    az, el = (
        spherical_coords[..., 1],
        spherical_coords[..., 2],
    )

    # Compute number of HEALPix pixels that cover the sphere
    n_pix = hp.nside2npix(nside)

    # Get HEALPix pixel indices for each event
    # HEALPix expects latitude in [-90, 90] so we don't need to change elevation
    hpix_idx = hp.ang2pix(nside, az, el, nest=nested, lonlat=True)

    # Initialize histogram: (n_HEALPix pixels, n_energy_bins)
    hist = np.zeros((n_pix, len(energy_bin_edges)))

    # Bin data in energy & HEALPix space
    for i, (e_min, e_max) in enumerate(energy_bin_edges):
        mask = (energy >= e_min) & (energy < e_max)
        # Only count the events that fall within the energy bin
        hist[:, i] += np.bincount(hpix_idx[mask], minlength=n_pix).astype(np.float64)

    return hist


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
    sc_exposure: np.ndarray,
) -> NDArray:
    """
    Compute a 3D array of the exposure in the helio frame.

    Parameters
    ----------
    time : np.ndarray
        Median time of pointing in J2000 seconds.
    sc_exposure : np.ndarray
        Spacecraft exposure.

    Returns
    -------
    exposure_3d : np.ndarray
        A 3D array with dimensions (az, el, energy).

    Notes
    -----
    These calculations are performed once per pointing.
    """
    # Get bins and midpoints, with angles in degrees.
    _, energy_midpoints = build_energy_bins()
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )

    # Initialize the exposure grid.
    exposure_3d = np.zeros(
        (len(el_bin_midpoints), len(az_bin_midpoints), len(energy_midpoints))
    )

    # Create a 3D Cartesian grid from spherical coordinates
    # using azimuth and elevation midpoints.
    az_grid, el_grid = np.meshgrid(az_bin_midpoints, el_bin_midpoints[::-1])

    # Radial distance.
    r = np.ones(el_grid.shape)
    spherical_coords = np.stack((r, az_grid, el_grid), axis=-1)
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    cartesian = cartesian_coords.reshape(-1, 3, order="F").T

    # Spacecraft velocity in the pointing (DPS) frame wrt heliosphere.
    state = imap_state(time, ref_frame=SpiceFrame.IMAP_DPS)

    # Extract the velocity part of the state vector
    spacecraft_velocity = state[3:6]

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
        helio_velocity = spacecraft_velocity.reshape(3, 1) + energy_velocity * cartesian

        # Normalized vectors representing the direction of the heliocentric velocity.
        helio_normalized = helio_velocity.T / np.linalg.norm(
            helio_velocity.T, axis=1, keepdims=True
        )
        # Converts vectors from Cartesian coordinates (x, y, z)
        # into spherical coordinates.
        spherical_coords = cartesian_to_spherical(helio_normalized)
        az, el = spherical_coords[..., 1], spherical_coords[..., 2]

        # Assign values from sc_exposure directly to bins.
        az_idx = np.digitize(az, az_bin_edges) - 1
        el_idx = np.digitize(el, el_bin_edges[::-1]) - 1

        # Ensure az_idx and el_idx are within bounds.
        az_idx = np.clip(az_idx, 0, len(az_bin_edges) - 2)
        el_idx = np.clip(el_idx, 0, len(el_bin_edges) - 2)

        # A 1D array of linear indices used to track the bin_id.
        idx = el_idx + az_idx * az_grid.shape[0]
        # Bins the transposed sc_exposure array.
        binned_exposure = sc_exposure.T.flatten(order="F")[idx]
        # Reshape the binned exposure.
        exposure_3d[:, :, i] = binned_exposure.reshape(az_grid.shape, order="F")

    return exposure_3d


def get_spacecraft_sensitivity(
    efficiencies: pandas.DataFrame,
    geometric_function: pandas.DataFrame,
) -> pandas.DataFrame:
    """
    Compute sensitivity.

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
    """
    # Exclude "Right Ascension (deg)" and "Declination (deg)" from the multiplication
    energy_columns = efficiencies.columns.difference(
        ["Right Ascension (deg)", "Declination (deg)"]
    )
    sensitivity = efficiencies[energy_columns].mul(
        geometric_function["Response (cm2-sr)"].values, axis=0
    )

    # Add "Right Ascension (deg)" and "Declination (deg)" to the result
    sensitivity.insert(
        0, "Right Ascension (deg)", efficiencies["Right Ascension (deg)"]
    )
    sensitivity.insert(1, "Declination (deg)", efficiencies["Declination (deg)"])

    return sensitivity
