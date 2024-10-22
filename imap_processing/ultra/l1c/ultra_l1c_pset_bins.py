"""Module to create bins for pointing sets."""

from pathlib import Path

import cdflib
import numpy as np
from numpy.typing import NDArray
import spiceypy as spice
import typing

from imap_processing.spice.kernels import ensure_spice
from imap_processing.ultra.constants import UltraConstants

# TODO: add species binning.


def build_energy_bins() -> tuple[np.ndarray, np.ndarray]:
    """
    Build energy bin boundaries.

    Returns
    -------
    energy_bin_edges : np.ndarray
        Array of energy bin edges.
    energy_midpoints : np.ndarray
        Array of energy bin midpoints.
    """
    # TODO: these value will almost certainly change.
    alpha = 0.2  # deltaE/E
    energy_start = 3.385  # energy start for the Ultra grids
    n_bins = 23  # number of energy bins

    # Calculate energy step
    energy_step = (1 + alpha / 2) / (1 - alpha / 2)

    # Create energy bins.
    energy_bin_edges = energy_start * energy_step ** np.arange(n_bins + 1)
    # Add a zero to the left side for outliers and round to nearest 3 decimal places.
    energy_bin_edges = np.around(np.insert(energy_bin_edges, 0, 0), 3)
    energy_midpoints = (energy_bin_edges[:-1] + energy_bin_edges[1:]) / 2

    return energy_bin_edges, energy_midpoints


def build_spatial_bins(
    spacing: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build spatial bin boundaries for azimuth and elevation.

    Parameters
    ----------
    spacing : float, optional
        The bin spacing in degrees (default is 0.5 degrees).

    Returns
    -------
    az_bin_edges : np.ndarray
        Array of azimuth bin boundary values.
    el_bin_edges : np.ndarray
        Array of elevation bin boundary values.
    az_bin_midpoints : np.ndarray
        Array of azimuth bin midpoint values.
    el_bin_midpoints : np.ndarray
        Array of elevation bin midpoint values.
    """
    # Azimuth bins from 0 to 360 degrees.
    az_bin_edges = np.arange(0, 360 + spacing, spacing)
    az_bin_midpoints = az_bin_edges[:-1] + spacing / 2  # Midpoints between edges

    # Elevation bins from -90 to 90 degrees.
    el_bin_edges = np.arange(-90, 90 + spacing, spacing)
    el_bin_midpoints = el_bin_edges[:-1] + spacing / 2  # Midpoints between edges

    return az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints


def cartesian_to_spherical(
    v: NDArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    v : np.ndarray
        A NumPy array with shape (n, 3) where each
        row represents a vector
        with x, y, z-components.

    Returns
    -------
    az : np.ndarray
        The azimuth angles in degrees.
    el : np.ndarray
        The elevation angles in degrees.
    r : np.ndarray
        The radii, or magnitudes, of the vectors.
    """
    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]

    # Magnitude of the velocity vector
    magnitude_v = np.sqrt(vx**2 + vy**2 + vz**2)

    vhat_x = -vx / magnitude_v
    vhat_y = -vy / magnitude_v
    vhat_z = -vz / magnitude_v

    # Elevation angle (angle from the z-axis, range: [-pi/2, pi/2])
    el = np.arcsin(vhat_z)

    # Azimuth angle (angle in the xy-plane, range: [0, 2*pi])
    az = np.arctan2(vhat_y, vhat_x)

    # Ensure azimuth is from 0 to 2PI
    az = az % (2 * np.pi)

    return np.degrees(az), np.degrees(el), magnitude_v


def spherical_to_cartesian(r: np.ndarray, theta: np.ndarray, phi: np.ndarray):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    r : np.ndarray
        Radius.
    theta : np.ndarray
        Azimuth angle in radians.
    phi : array-like or float
        Elevation angle in radians.

    Returns:
    x, y, z : tuple
        Cartesian coordinates.
    """
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)

    return x, y, z


def get_histogram(
    v: tuple[np.ndarray, np.ndarray, np.ndarray],
    energy: np.ndarray,
    az_bin_edges: np.ndarray,
    el_bin_edges: np.ndarray,
    energy_bin_edges: np.ndarray,
) -> NDArray:
    """
    Compute a 3D histogram of the particle data.

    Parameters
    ----------
    v : tuple[np.ndarray, np.ndarray, np.ndarray]
        The x,y,z-components of the velocity vector.
    energy : np.ndarray
        The particle energy.
    az_bin_edges : np.ndarray
        Array of azimuth bin boundary values.
    el_bin_edges : np.ndarray
        Array of elevation bin boundary values.
    energy_bin_edges : np.ndarray
        Array of energy bin edges.

    Returns
    -------
    hist : np.ndarray
        A 3D histogram array.
    """
    az, el, _ = cartesian_to_spherical(v)

    # 3D binning.
    hist, _ = np.histogramdd(
        sample=(az, el, energy), bins=[az_bin_edges, el_bin_edges, energy_bin_edges]
    )

    return hist


def get_pointing_frame_exposure_times(
    constant_exposure: Path, n_spins: int, sensor: str
) -> NDArray:
    """
    Compute a 2D array of the exposure.

    Parameters
    ----------
    constant_exposure : Path
        Path to file containing constant exposure data.
    n_spins : int
        Number of spins per pointing.
    sensor : str
        Sensor (45 or 90).

    Returns
    -------
    exposure : np.ndarray
        A 2D array with dimensions (az, el).
    """
    with cdflib.CDF(constant_exposure) as cdf_file:
        exposure = cdf_file.varget(f"dps_grid{sensor}") * n_spins

    return exposure


@ensure_spice
@typing.no_type_check
def get_helio_exposure_times(
    time: np.ndarray,
    sc_exposure: np.ndarray,
) -> np.ndarray:
    """
    Compute a 3D array of the exposure in the helio frame.

    Parameters
    ----------
    time : np.ndarray
        Median time of pointing.
    sc_exposure : np.ndarray
        Spacecraft exposure.

    Returns
    -------
    exposure_3d : np.ndarray
        A 3D array with dimensions (az, el, energy).

    Notes
    -------
    These calculations are performed once per pointing.
    """
    # Get bins and midpoints.
    energy_bin_edges, energy_midpoints = build_energy_bins()
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )
    # Initialize the exposure grid.
    exposure_3d = np.zeros((len(el_bin_midpoints),
                            len(az_bin_midpoints),
                            len(energy_midpoints)))

    # Create a 3D Cartesian grid from spherical coordinates
    # using azimuth and elevation midpoints.
    az_grid, el_grid =  np.meshgrid(az_bin_midpoints, el_bin_midpoints[::-1])

    # Radial distance.
    r = np.ones(el_grid.shape)
    x, y, z = spherical_to_cartesian(r,
                                     np.radians(az_grid),
                                     np.radians(el_grid))

    # Reshape and combine the Cartesian coordinates into a 2D array.
    cartesian = np.vstack([x.flatten(order='F'),
                       y.flatten(order='F'),
                       z.flatten(order='F')])

    # Spacecraft velocity in the pointing (DPS) frame wrt heliosphere.
    state, lt = spice.spkezr("IMAP", time, "IMAP_DPS", "NONE", "SUN")

    # Extract the velocity part of the state vector
    spacecraft_velocity = state[3:6]

    for i, energy_midpoint in enumerate(energy_midpoints):
        # Convert the midpoint energy to a velocity (km/s).
        # Based on kinetic energy equation: E = 1/2 * m * v^2.
        energy_velocity = np.sqrt(2 * energy_midpoint * UltraConstants.KEV_J / UltraConstants.MASS_H) / 1e3

        # Use Compton-Getting to transform the velocity wrt spacecraft
        # to the velocity wrt heliosphere.
        # energy_velocity * cartesian -> apply the magnitude of the velocity
        # to every position on the grid in the despun grid.
        helio_velocity = spacecraft_velocity.reshape(3, 1) + \
                         energy_velocity * cartesian

        # Normalized vectors representing the direction of the heliocentric velocity.
        helio_normalized = helio_velocity.T / np.linalg.norm(helio_velocity.T, axis=1, keepdims=True)
        # Converts vectors from Cartesian coordinates (x, y, z)
        # into spherical coordinates
        az, el, _ = cartesian_to_spherical(-helio_normalized)

        # Bin the coordinates.
        az_idx = np.digitize(az, az_bin_edges) - 1
        el_idx = np.digitize(el, el_bin_edges[::-1]) - 1

        # Ensure az_idx and el_idx are within bounds.
        az_idx = np.clip(az_idx, 0, len(az_bin_edges) - 2)
        el_idx = np.clip(el_idx, 0, len(el_bin_edges) - 2)

        # A 1D array of linear indices used to track the bin_id.
        idx = el_idx + az_idx * az_grid.shape[0]
        # Bins the transposed sc_exposure array.
        binned_exposure = sc_exposure.T.flatten(order='F')[idx]
        # Reshape the binned exposure.
        exposure_3d[:, :, i] = binned_exposure.reshape(az_grid.shape, order='F')

    return exposure_3d


def get_pointing_frame_sensitivity(
        constant_sensitivity: Path,
        n_spins: int, sensor: str
) -> NDArray:
    """
    Compute a 3D array of the sensitivity.

    Parameters
    ----------
    constant_sensitivity : Path
        Path to file containing constant sensitivity data.
    n_spins : int
        Number of spins per pointing.
    sensor : str
        Sensor (45 or 90).

    Returns
    -------
    sensitivity : np.ndarray
        A 3D array with dimensions (az, el, energy).
    """
    with cdflib.CDF(constant_sensitivity) as cdf_file:
        sensitivity = cdf_file.varget(f"dps_sensitivity{sensor}") * n_spins

    return sensitivity
