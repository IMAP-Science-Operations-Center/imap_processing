"""Module to create pointing sets."""

from pathlib import Path

import cdflib
import numpy as np
from numpy.typing import NDArray

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


def build_spatial_bins(
    az_spacing: float = 0.5,
    el_spacing: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build spatial bin boundaries for azimuth and elevation.

    Parameters
    ----------
    az_spacing : float, optional
        The azimuth bin spacing in degrees (default is 0.5 degrees).
    el_spacing : float, optional
        The elevation bin spacing in degrees (default is 0.5 degrees).

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
    az_bin_edges = np.arange(0, 360 + az_spacing, az_spacing)
    az_bin_midpoints = az_bin_edges[:-1] + az_spacing / 2  # Midpoints between edges

    # Elevation bins from -90 to 90 degrees.
    el_bin_edges = np.arange(-90, 90 + el_spacing, el_spacing)
    el_bin_midpoints = el_bin_edges[:-1] + el_spacing / 2  # Midpoints between edges

    return az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints


def get_histogram(
    vhat: tuple[np.ndarray, np.ndarray, np.ndarray],
    energy: np.ndarray,
    az_bin_edges: np.ndarray,
    el_bin_edges: np.ndarray,
    energy_bin_edges: list[tuple[float, float]],
) -> NDArray:
    """
    Compute a 3D histogram of the particle data.

    Parameters
    ----------
    vhat : tuple[np.ndarray, np.ndarray, np.ndarray]
        The x,y,z-components of the unit velocity vector.
    energy : np.ndarray
        The particle energy.
    az_bin_edges : np.ndarray
        Array of azimuth bin boundary values.
    el_bin_edges : np.ndarray
        Array of elevation bin boundary values.
    energy_bin_edges : list[tuple[float, float]]
        Array of energy bin edges.

    Returns
    -------
    hist : np.ndarray
        A 3D histogram array.

    Notes
    -----
    The histogram will now work properly for overlapping energy bins, i.e.
    the same energy value can fall into multiple bins if the intervals overlap.
    """
    spherical_coords = cartesian_to_spherical(vhat)
    az, el = (
        spherical_coords[..., 1],
        spherical_coords[..., 2],
    )

    # Initialize histogram
    hist_total = np.zeros(
        (len(az_bin_edges) - 1, len(el_bin_edges) - 1, len(energy_bin_edges))
    )

    for i, (e_min, e_max) in enumerate(energy_bin_edges):
        # Filter data for current energy bin.
        mask = (energy >= e_min) & (energy < e_max)
        hist, _ = np.histogramdd(
            sample=(az[mask], el[mask], energy[mask]),
            bins=[az_bin_edges, el_bin_edges, [e_min, e_max]],
        )
        # Assign 2D histogram to current energy bin.
        hist_total[:, :, i] = hist[:, :, 0]

    return hist_total


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
    # Get bins and midpoints.
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
    spherical_coords = np.stack((r, np.radians(az_grid), np.radians(el_grid)), axis=-1)
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


def get_pointing_frame_sensitivity(
    constant_sensitivity: Path, n_spins: int, sensor: str
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
