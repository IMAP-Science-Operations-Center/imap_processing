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

    return energy_bin_start, energy_bin_end


def build_spatial_bins(spacing: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
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
    """
    # Azimuth bins from 0 to 360 degrees
    az_bin_edges = np.arange(0, 360 + spacing, spacing)

    # Elevation bins from -90 to 90 degrees
    el_bin_edges = np.arange(-90, 90 + spacing, spacing)

    return az_bin_edges, el_bin_edges
