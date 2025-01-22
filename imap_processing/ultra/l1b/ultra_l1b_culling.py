"""Culls Events for ULTRA L1b."""
# TODO: Add "bad attitude times" to the culling process.
# TODO: Implement threshold calculations.
# TODO: Add rates data.

import numpy as np
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapHkUltraFlags
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
    quality_flags = np.full(
        hist.shape[0] * hist.shape[1], ImapHkUltraFlags.NONE.value, dtype=np.uint16
    )

    appended_spin = np.empty(0, dtype=np.uint16)
    appended_energy = np.empty(0, dtype=np.float64)

    for energy_idx in range(hist.shape[0]):
        # Count rates for each spin at this energy
        spin_count_rates = hist[energy_idx][:]
        # Indices where the counts exceed the threshold
        indices = np.nonzero(
            spin_count_rates > UltraConstants.COUNT_RATES_THRESHOLDS[energy_idx]
        )
        flattened_indices = energy_idx * hist.shape[1] + indices[0]
        quality_flags[flattened_indices] |= ImapHkUltraFlags.HIGHCOUNTS.value

        # Calculate the energy midpoint for each bin
        energy_midpoint = (
            UltraConstants.CULLING_ENERGY_BIN_EDGES[energy_idx]
            + UltraConstants.CULLING_ENERGY_BIN_EDGES[energy_idx + 1]
        ) / 2

        appended_spin = np.concatenate((appended_spin, np.unique(spin)))
        appended_energy = np.concatenate(
            (appended_energy, np.full(len(np.unique(spin)), energy_midpoint))
        )

    return quality_flags, appended_spin, appended_energy
