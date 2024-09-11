"""Module to create bins for pointing sets."""

import numpy as np


def build_energy_bins() -> tuple[np.ndarray, np.ndarray]:
    """
    Build energy bin boundaries.

    Returns
    -------
    energy_bin_start : np.ndarray
        Array of energy bin start values.
    energy_bin_end : np.ndarray
        Array of energy bin end values.
    """
    alpha = 0.05  # deltaE/E
    energy_start = 3.5  # energy start for the Ultra grids
    n_bins = 90  # number of energy bins

    # Calculate energy step
    energy_step = (1 + alpha / 2) / (1 - alpha / 2)

    # Create energy bins.
    bin_edges = energy_start * energy_step ** np.arange(n_bins + 1)
    energy_bin_start = bin_edges[:-1]
    energy_bin_end = bin_edges[1:]

    return energy_bin_start, energy_bin_end, az_bin_midpoints, el_bin_midpoints


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
    vx_dps_sc: np.ndarray, vy_dps_sc: np.ndarray, vz_dps_sc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    vx_dps_sc : np.ndarray
        The x-components of the velocity vector.
    vy_dps_sc : np.ndarray
        The y-components of the velocity vector.
    vz_dps_sc : np.ndarray
        The z-components of the velocity vector.

    Returns
    -------
    az : np.ndarray
        The azimuth angles in degrees.
    el : np.ndarray
        The elevation angles in degrees.
    """
    # Magnitude of the velocity vector
    magnitude_v = np.sqrt(vx_dps_sc**2 + vy_dps_sc**2 + vz_dps_sc**2)

    vhat_x = -vx_dps_sc / magnitude_v
    vhat_y = -vy_dps_sc / magnitude_v
    vhat_z = -vz_dps_sc / magnitude_v

    # Convert from cartesian to spherical coordinates (azimuth, elevation)
    # Radius (magnitude)
    r = np.sqrt(vhat_x**2 + vhat_y**2 + vhat_z**2)

    # Elevation angle (angle from the z-axis, range: [-pi/2, pi/2])
    el = np.arcsin(vhat_z / r)

    # Azimuth angle (angle in the xy-plane, range: [0, 2*pi])
    az = np.arctan2(vhat_y, vhat_x)

    # Ensure azimuth is from 0 to 2PI
    az = az % (2 * np.pi)

    return az, el


def bin_space(vx_dps_sc, vy_dps_sc, vz_dps_sc) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin particle.

    Parameters
    ----------
    vx_dps_sc : float, optional
        The bin spacing in degrees (default is 0.5 degrees).
    vy_dps_sc : float, optional
        The bin spacing in degrees (default is 0.5 degrees).
    vz_dps_sc : float, optional
        The bin spacing in degrees (default is 0.5 degrees).

    Returns
    -------
    az_midpoint : np.ndarray
        Array of azimuth bin boundary values.
    el_midpoint : np.ndarray
        Array of elevation bin boundary values.
    """
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )

    az, el = cartesian_to_spherical(vx_dps_sc, vy_dps_sc, vz_dps_sc)

    # Find the appropriate bin index using searchsorted
    az_bin_idx = np.searchsorted(az_bin_edges, np.degrees(az), side="right") - 1
    el_bin_idx = np.searchsorted(el_bin_edges, np.degrees(el), side="right") - 1

    # Assign the corresponding midpoints
    az_midpoint = az_bin_midpoints[az_bin_idx]
    el_midpoint = el_bin_midpoints[el_bin_idx]

    return az_midpoint, el_midpoint
