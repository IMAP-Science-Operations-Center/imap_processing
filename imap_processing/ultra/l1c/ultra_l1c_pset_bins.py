"""Module to create bins for pointing sets."""

from pathlib import Path

import cdflib
import numpy as np
from numpy.typing import NDArray

# TODO: add species binning.


def build_energy_bins() -> NDArray[np.float64]:
    """
    Build energy bin boundaries.

    Returns
    -------
    energy_bin_edges : np.ndarray
        Array of energy bin edges.
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

    return energy_bin_edges


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
