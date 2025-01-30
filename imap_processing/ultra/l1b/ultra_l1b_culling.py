"""Culls Events for ULTRA L1b."""

import numpy as np
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags
from imap_processing.spice.spin import get_spin_data, interpolate_spin_data
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
    spin_df = interpolate_spin_data(eventtimes_met)
    return spin_df["spin_number"].values


def get_energy_histogram(
    spin_number: NDArray, energy: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
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
    counts : NDArray
        A 2D histogram array containing the
        counts per spin at each energy bin.
    """
    spin_df = get_spin_data()

    spin_edges = np.unique(spin_number)
    spin_edges = np.append(spin_edges, spin_edges.max() + 1)

    # Counts per spin at each energy bin.
    hist, _ = np.histogramdd(
        sample=(energy, spin_number),
        bins=[UltraConstants.CULLING_ENERGY_BIN_EDGES, spin_edges],
    )

    counts = hist.copy()

    # Count rate per spin at each energy bin.
    for i in range(hist.shape[1]):
        spin_duration = spin_df.spin_period_sec[spin_df.spin_number == i]
        hist[:, i] /= spin_duration.values[0]

    return hist, spin_edges, counts


def flag_attitude(eventtimes_met: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Flag data based on attitude.

    Parameters
    ----------
    eventtimes_met : NDArray
        Event Times in Mission Elapsed Time.

    Returns
    -------
    quality_flags : NDArray
        Quality flags.
    spin_rates : NDArray
        Spin rates.
    spin_period : NDArray
        Spin period.
    spin_starttime : NDArray
        Spin start time.
    """
    spins = np.unique(get_spin(eventtimes_met))  # Get unique spins
    spin_df = get_spin_data()  # Load spin data

    spin_period = spin_df.loc[spin_df.spin_number.isin(spins), "spin_period_sec"]
    spin_starttime = spin_df.loc[spin_df.spin_number.isin(spins), "spin_start_time"]
    spin_rates = 60 / spin_period  # 60 seconds in a minute
    indices = spin_rates > np.array(UltraConstants.RPM)

    quality_flags = np.full(
        spin_rates.shape, ImapAttitudeUltraFlags.NONE.value, dtype=np.uint16
    )
    quality_flags[indices] |= ImapAttitudeUltraFlags.SPINRATE.value

    return quality_flags, spin_rates, spin_period, spin_starttime


def get_n_sigma(counts: NDArray, sigma: int = 6) -> NDArray:
    """
    Calculate n sigma.

    Parameters
    ----------
    counts : NDArray
        A 2D histogram array containing the
        counts per spin at each energy bin.
    sigma : int (default=6)
        The number of sigma.

    Returns
    -------
    six_sigma_per_energy : NDArray
        Six sigma per energy.
    """
    sigma_per_energy = np.std(counts, axis=1)
    n_sigma_per_energy = sigma * sigma_per_energy

    return n_sigma_per_energy


def flag_spin(
    eventtimes_met: NDArray, energy: NDArray, sigma: int = 6
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Flag data based on counts and negative energies.

    Parameters
    ----------
    eventtimes_met : NDArray
        Event Times in Mission Elapsed Time.
    energy : NDArray
        Energy data.
    sigma : int (default=6)
        The number of sigma.

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
    hist, spin_edges, counts = get_energy_histogram(spin, energy)
    n_sigma_per_energy = get_n_sigma(counts, sigma=sigma)
    quality_flags = np.full(hist.shape, ImapRatesUltraFlags.NONE.value, dtype=np.uint16)

    bin_edges = np.array(UltraConstants.CULLING_ENERGY_BIN_EDGES)
    energy_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    spin = np.unique(spin)

    # Indices where the counts exceed the threshold
    indices = hist > np.array(UltraConstants.COUNT_RATES_THRESHOLDS)[:, np.newaxis]
    quality_flags[indices] |= ImapRatesUltraFlags.HIGHCOUNTS.value

    indices_n_sigma = counts > n_sigma_per_energy[:, np.newaxis]
    quality_flags[indices_n_sigma] |= ImapRatesUltraFlags.SIXSIGMA.value

    return quality_flags, spin, energy_midpoints
