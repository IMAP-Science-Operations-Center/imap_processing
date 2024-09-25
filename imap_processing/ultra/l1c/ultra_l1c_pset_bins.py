"""Module to create bins for pointing sets."""

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
    vx, vy, vz = np.hsplit(v, 3)
    vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()  # Flatten the arrays

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

    return np.degrees(az), np.degrees(el), r


def extract_non_zero_indices_and_counts(
    hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract indices and counts of non-zero elements from a histogram.

    Parameters
    ----------
    hist : np.ndarray
        3D histogram with counts for azimuth, elevation, and energy bins.

    Returns
    -------
    non_zero_indices : np.ndarray
        Array of indices of non-zero elements in the 3D histogram.
    non_zero_counts : np.ndarray
        Array of non-zero counts from the 3D histogram.
    """
    non_zero_indices = np.argwhere(hist > 0).T
    non_zero_counts = hist[hist > 0]
    return non_zero_indices, non_zero_counts


def get_histogram(
    v: tuple[np.ndarray, np.ndarray, np.ndarray],
    energy: np.ndarray,
    az_bin_edges: np.ndarray,
    el_bin_edges: np.ndarray,
    energy_bin_edges: np.ndarray,
) -> tuple[NDArray, NDArray]:
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


def create_unique_identifiers(
    hist: np.ndarray,
    az_bin_midpoints: np.ndarray,
    el_bin_midpoints: np.ndarray,
) -> tuple:
    """
    Create unique identifiers for spatial bins and generate non-zero counts.

    Parameters
    ----------
    hist : np.ndarray
        3D histogram with counts for azimuth, elevation, and energy bins.
    az_bin_midpoints : np.ndarray
        Midpoints of the azimuth bins.
    el_bin_midpoints : np.ndarray
        Midpoints of the elevation bins.

    Returns
    -------
    counts : np.ndarray
        A 2D array containing the binned counts for each spatial and energy bin.
    bin_id : tuple[np.ndarray, np.ndarray]
        Unique identifiers for the spatial and energy bins.
    midpoints : tuple[np.ndarray, np.ndarray]
        Azimuth and elevation midpoints for the spatial bins.
    """
    non_zero_indices, non_zero_counts = extract_non_zero_indices_and_counts(hist)

    # Extract the indices of all non-zero elements in the 3D hist array
    az_bin_id, el_bin_id, energy_bin_id = non_zero_indices

    # Unique identifier for spatial bins by combining azimuth and elevation indices
    bin_id = (np.unique(az_bin_id), np.unique(el_bin_id), np.unique(energy_bin_id))

    # Get unique azimuth and elevation midpoints
    midpoints = (
        az_bin_midpoints[bin_id[0]],
        el_bin_midpoints[bin_id[1]],
    )

    # Determine positions in unique coordinates for assigning counts
    az_positions = np.searchsorted(bin_id[0], az_bin_id)
    el_positions = np.searchsorted(bin_id[1], el_bin_id)
    energy_positions = np.searchsorted(bin_id[2], energy_bin_id)

    counts = np.zeros((len(bin_id[0]), len(bin_id[1]), len(bin_id[2])))
    counts[az_positions, el_positions, energy_positions] = non_zero_counts

    return counts, bin_id, midpoints


def bin_data(v: tuple[np.ndarray, np.ndarray, np.ndarray], energy: np.ndarray) -> tuple:
    """
    Bin the data spatially and by energy.

    Parameters
    ----------
    v : tuple[np.ndarray, np.ndarray, np.ndarray]
        A 2D array containing the x, y, z components of velocity vector.
    energy : np.ndarray
        A 1D array with shape (N,) containing the particle energy.

    Returns
    -------
    counts : np.ndarray
        A 2D array containing the binned counts for each spatial and energy bin.
    bin_id : tuple[np.ndarray, np.ndarray]
        Unique identifiers for the spatial and energy bins.
    midpoints : tuple[np.ndarray, np.ndarray]
        Azimuth and elevation midpoints for the spatial bins.
    energy_edge_start : np.ndarray
        The starting edges of the energy bins corresponding to non-zero counts.

    Notes
    -----
    The returned data can be used to construct a xarray Dataset as follows:

    ds = xr.Dataset(
        {
            "az_midpoint": ("az_bin_id", midpoints[0]),
            "el_midpoint": ("el_bin_id", midpoints[1]),
            "energy_edge_start": ("energy_bin_id", energy_edge_start),
            "counts": (("az_bin_id", "el_bin_id","energy_bin_id"), counts),
        },
        coords={
            "az_bin_id": bin_id[0],
            "el_bin_id": bin_id[1],
            "energy_bin_id": bin_id[2],
        }
    )
    """
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )
    energy_bin_edges = build_energy_bins()

    # Compute the 3D histogram of the particle data
    hist = get_histogram(v, energy, az_bin_edges, el_bin_edges, energy_bin_edges)

    # Get the starting edges for energy bins
    energy_bin_start = energy_bin_edges[:-1]

    _, non_zero_counts = extract_non_zero_indices_and_counts(hist)

    # Create unique identifiers and map non-zero counts to 2D histogram array
    counts, bin_id, midpoints = create_unique_identifiers(
        hist, az_bin_midpoints, el_bin_midpoints, non_zero_counts
    )

    _, _, energy_bin = np.argwhere(hist > 0).T
    energy_edge_start = energy_bin_start[energy_bin]

    return counts, bin_id, midpoints, energy_edge_start
