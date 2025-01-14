"""Culls Events for ULTRA L1b."""

import numpy as np
from numpy.typing import NDArray
import xarray

from imap_processing.spice.geometry import get_spin_data
from imap_processing.quality_flags import ImapUltraFlags
from imap_processing.ultra.constants import UltraConstants


def get_spin(met: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    Get spin parameters for each event.

    Parameters
    ----------
    met : NDArray
        Mission Elaspsed Time.

    Returns
    -------
    spin_number : NDArray
        Spin number from Universal Spin Table.
    spin_start_time : NDArray
        Spin start time from Universal Spin Table.
    spin_duration : NDArray
        Spin duration from Universal Spin Table.
    """
    spin_df = get_spin_data()

    last_spin_indices = (
        np.searchsorted(spin_df["spin_start_time"], met, side="right") - 1
    )
    spin_number = spin_df["spin_number"].values[last_spin_indices]
    spin_start_time = spin_df["spin_start_time"].values[last_spin_indices]
    spin_duration = spin_df["spin_period_sec"].values[last_spin_indices]

    return spin_number, spin_start_time, spin_duration


def get_energy_histogram(spin_number: NDArray,
                         energy: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute a 2D histogram of the counts.

    Parameters
    ----------
    spin_number : NDArray
        Spin number.
    energy : NDArray
        The particle energy.

    Returns
    -------
    hist : NDArray
        A 2D histogram array.
    spin_edges : NDArray
        Edges of the spin number bins.
    """
    spin_edges = np.unique(spin_number)
    spin_edges = np.append(spin_edges, spin_edges[-1] + 1)

    # 2D binning.
    hist, _ = np.histogramdd(sample=(energy, spin_number),
                             bins=[UltraConstants.CULLING_ENERGY_BINS,
                                   spin_edges])

    return hist, spin_edges


def flag_spin(met: NDArray, energy: NDArray)-> NDArray:
    """
    Flags data based on counts and negative energies.

    Parameters
    ----------
    l1b_de_dataset : xarray.Dataset
        Data in xarray format.

    Returns
    -------
    quality_flags_data : NDArray
        Quality flags.
    """
    quality_flags_data = np.zeros(len(met), np.uint16)

    # Flag negative energies.
    quality_flags_data[energy < 0] |= ImapUltraFlags.NEG.value

    spin = get_spin(met)
    hist, spin_edges, energy_bin_edges = get_energy_histogram(spin, energy)

    # Map data points to bins
    energy_bin_idx = np.digitize(energy, bins=energy_bin_edges) - 1
    spin_bin_idx = np.digitize(spin, bins=spin_edges) - 1

    # Define thresholds for each energy bin
    thresholds = [
        UltraConstants.COUNTS_THRESHOLD_0_10_KEV,  # For energy range -1e5 to 0-10 keV
        UltraConstants.COUNTS_THRESHOLD_10_20_KEV,  # For energy range 10-20 keV
        UltraConstants.COUNTS_THRESHOLD_GE_20_KEV,  # For energy range >=20 keV
    ]

    # Iterate through non-empty bins
    for energy_idx, spin_idx in np.argwhere(hist > 0):
        threshold = thresholds[min(energy_idx, len(thresholds) - 1)]
        if hist[energy_idx, spin_idx] > threshold:
            # Create a mask for data points in the current high-count bin
            mask = (energy_bin_idx == energy_idx) & (spin_bin_idx == spin_idx)
            quality_flags_data[mask] |= ImapUltraFlags.BADSPIN.value

    return quality_flags_data
