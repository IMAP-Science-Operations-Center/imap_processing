"""Module to create energy bins for pointing sets."""

import numpy as np
from numpy.typing import NDArray


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
    v: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    v : tuple[np.ndarray, np.ndarray, np.ndarray]
        The x,y,z-components of the velocity vector.

    Returns
    -------
    az : np.ndarray
        The azimuth angles in degrees.
    el : np.ndarray
        The elevation angles in degrees.
    r : np.ndarray
        The radii, or magnitudes, of the vectors.
    """
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    # Magnitude of the velocity vector
    magnitude_v = np.sqrt(vx**2 + vy**2 + vz**2)

    vhat_x = -vx / magnitude_v
    vhat_y = -vy / magnitude_v
    vhat_z = -vz / magnitude_v

    # Convert from cartesian to spherical coordinates (azimuth, elevation)
    # Radius (magnitude)
    r = np.sqrt(vhat_x**2 + vhat_y**2 + vhat_z**2)

    # Elevation angle (angle from the z-axis, range: [-pi/2, pi/2])
    el = np.arcsin(vhat_z / r)

    # Azimuth angle (angle in the xy-plane, range: [0, 2*pi])
    az = np.arctan2(vhat_y, vhat_x)

    # Ensure azimuth is from 0 to 2PI
    az = az % (2 * np.pi)

    return az, el, r


def bin_space(
    v: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Bin the particle.

    Parameters
    ----------
    v : tuple[np.ndarray, np.ndarray, np.ndarray]
        The x,y,z-components of the velocity vector.

    Returns
    -------
    az_midpoint : np.ndarray
        Array of azimuth midpoint values.
    el_midpoint : np.ndarray
        Array of elevation midpoint values.
    """
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )

    az, el, _ = cartesian_to_spherical(v)

    az_degrees = np.degrees(az)
    el_degrees = np.degrees(el)

    # If azimuth is exactly 360 degrees it is placed in last bin.
    az_degrees[az_degrees == az_bin_edges[-1]] = az_bin_midpoints[-1]
    # If elevation is exactly 90 degrees it is placed in last bin.
    el_degrees[el_degrees == el_bin_edges[-1]] = el_bin_midpoints[-1]

    # Find the appropriate bin index.
    az_bin_idx = np.searchsorted(az_bin_edges, az_degrees, side="right") - 1
    el_bin_idx = np.searchsorted(el_bin_edges, el_degrees, side="right") - 1

    bin_id = az_bin_idx * len(el_bin_midpoints) + el_bin_idx
    _, counts = np.unique(bin_id, return_counts=True)

    return bin_id


def bin_energy(energy: np.ndarray) -> NDArray[np.float64]:
    """
    Bin the particle.

    Parameters
    ----------
    energy : np.ndarray
        Particle energy.

    Returns
    -------
    energy_start : np.ndarray
        Start of energy bin.
    """
    energy_bin_edges = build_energy_bins()

    # Find the appropriate bin index.
    energy_bin_idx = np.searchsorted(energy_bin_edges, energy, side="right") - 1

    return energy_bin_edges[energy_bin_idx]
