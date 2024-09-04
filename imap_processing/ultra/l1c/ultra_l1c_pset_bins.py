"""Module to create energy bins for pointing sets."""

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
    energy_bounds = [3.5, 300]  # energy bounds for the Ultra grids

    # Calculate energy step
    energy_step = (1 + alpha / 2) / (1 - alpha / 2)

    # Create bins.
    energy_bin_start = np.array([energy_bounds[0]])
    while energy_bin_start[-1] * energy_step <= energy_bounds[-1]:
        energy_bin_start = np.append(
            energy_bin_start, energy_bin_start[-1] * energy_step
        )

    # Compute the end values for the bins
    energy_bin_end = energy_bin_start[1:] * energy_step
    energy_bin_end = np.insert(energy_bin_end, 0, energy_bounds[0] * energy_step)

    return energy_bin_start, energy_bin_end
