"""Module to create pointing sets."""

from pathlib import Path

import cdflib
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from imap_processing.ena_maps.utils.spatial_utils import build_spatial_bins
from imap_processing.spice.geometry import (
    SpiceFrame,
    cartesian_to_spherical,
    imap_state,
    spherical_to_cartesian,
)
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ena_maps.utils.map_utils import bin_single_array_at_indices

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


def get_histogram(
    vhat: tuple[np.ndarray, np.ndarray, np.ndarray],
    energy: np.ndarray,
    energy_bin_edges: list[tuple[float, float]],
    nside: int = 32,
    nested: bool = True,
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
    nside : int
        The nside parameter of the Healpix tessellation.
    nested : bool, optional
        Whether the Healpix tessellation is nested. Default is False.

    Returns
    -------
    hist_total : np.ndarray
        A 3D histogram array with shape (n_pix, n_energy_bins).
    """
    spherical_coords = cartesian_to_spherical(vhat, degrees=True)
    az, el = spherical_coords[..., 1], spherical_coords[..., 2]

    # Convert elevation to HEALPix-compatible latitude
    lat = np.degrees(90 - el)  # Ensure lat is degrees

    # Compute number of HEALPix pixels
    n_pix = hp.nside2npix(nside)

    # Get HEALPix pixel indices
    hpix_idx = hp.ang2pix(nside, np.degrees(az), lat, nest=nested, lonlat=True)

    # Initialize histogram: (n_HEALPix pixels, n_energy_bins)
    hist_total = np.zeros((n_pix, len(energy_bin_edges)), dtype=np.float64)

    # Bin data in energy & HEALPix space
    for i, (e_min, e_max) in enumerate(energy_bin_edges):
        mask = (energy >= e_min) & (energy < e_max)

        hist_total[:, i] = bin_single_array_at_indices(
            value_array=np.ones(mask.sum(), dtype=np.float64),  # Count occurrences
            projection_grid_shape=(n_pix,),
            projection_indices=hpix_idx[mask],
        )

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
