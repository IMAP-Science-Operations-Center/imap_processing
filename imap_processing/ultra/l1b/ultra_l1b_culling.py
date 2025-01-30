"""Culls Events for ULTRA L1b."""
# TODO: Add "bad attitude times" to the culling process.
# TODO: Implement threshold calculations.
# TODO: Add rates data.

import numpy as np
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapRatesUltraFlags
from imap_processing.spice.geometry import get_spin_data
from imap_processing.ultra.constants import UltraConstants


def get_spin(eventtimes_met: NDArray) -> NDArray:
    """
    Get spin number for each event.

    Parameters
    ----------
    eventtimes_met : NDArray
        Event Times in Mission Elapsed Time.

    Returns
    -------
    spin_number : NDArray
        Spin number at each event derived the from Universal Spin Table.
    """
    spin_df = get_spin_data()

    last_spin_indices = (
        np.searchsorted(spin_df["spin_start_time"], eventtimes_met, side="right") - 1
    )
    spin_number = spin_df["spin_number"].values[last_spin_indices]

    return spin_number


def get_energy_histogram(
    spin_number: NDArray, energy: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Compute a 2D histogram of the counts binned by energy and spin number.

    Parameters
    ----------
    spin_number : NDArray
        Spin number.
    energy : NDArray
        The particle energy.

    Returns
    -------
    hist : NDArray
        A 2D histogram array containing the
        count rate per spin at each energy bin.
    spin_edges : NDArray
        Edges of the spin number bins.
    """
    spin_df = get_spin_data()

    spin_edges = np.unique(spin_number)
    spin_edges = np.append(spin_edges, spin_edges.max() + 1)

    # Counts per spin at each energy bin.
    hist, _ = np.histogramdd(
        sample=(energy, spin_number),
        bins=[UltraConstants.CULLING_ENERGY_BIN_EDGES, spin_edges],
    )

    # Count rate per spin at each energy bin.
    for i in range(hist.shape[1]):
        spin_duration = spin_df.spin_period_sec[spin_df.spin_number == i]
        hist[:, i] /= spin_duration.values[0]

    return hist, spin_edges


def flag_spin(
    eventtimes_met: NDArray, energy: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Flag data based on counts and negative energies.

    Parameters
    ----------
    eventtimes_met : NDArray
        Event Times in Mission Elapsed Time.
    energy : NDArray
        Energy data.

    Returns
    -------
    quality_flags : NDArray
        Quality flags.
    appended_spin : NDArray
        Spin data.
    appended_energy : NDArray
        Energy midpoint data.
    """
    spin = get_spin(eventtimes_met)
    hist, spin_edges = get_energy_histogram(spin, energy)
    quality_flags = np.full(hist.shape, ImapRatesUltraFlags.NONE.value, dtype=np.uint16)

    bin_edges = np.array(UltraConstants.CULLING_ENERGY_BIN_EDGES)
    energy_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    spin = np.unique(spin)

    # Indices where the counts exceed the threshold
    indices = hist > np.array(UltraConstants.COUNT_RATES_THRESHOLDS)[:, np.newaxis]
    quality_flags[indices] |= ImapRatesUltraFlags.HIGHCOUNTS.value

    return quality_flags, spin, energy_midpoints
