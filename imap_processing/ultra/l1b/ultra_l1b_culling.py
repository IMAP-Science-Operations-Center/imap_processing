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
) -> tuple[NDArray, NDArray, NDArray, float]:
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
    mean_duration : float
        Mean duration of the spin.
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
    total_spin_duration = 0

    # Count rate per spin at each energy bin.
    for i in range(hist.shape[1]):
        spin_duration = spin_df.spin_period_sec[spin_df.spin_number == i]
        hist[:, i] /= spin_duration.values[0]
        total_spin_duration += spin_duration.sum()

    mean_duration = total_spin_duration / hist.shape[1]

    return hist, spin_edges, counts, mean_duration


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
    bad_spin_rate_indices = (spin_rates < UltraConstants.CULLING_RPM_MIN) | (
        spin_rates > UltraConstants.CULLING_RPM_MAX
    )

    quality_flags = np.full(
        spin_rates.shape, ImapAttitudeUltraFlags.NONE.value, dtype=np.uint16
    )
    quality_flags[bad_spin_rate_indices] |= ImapAttitudeUltraFlags.SPINRATE.value

    return quality_flags, spin_rates, spin_period, spin_starttime


def get_n_sigma(count_rates: NDArray, mean_duration: float, sigma: int = 6) -> NDArray:
    """
    Calculate the threshold for the HIGHRATES flag.

    Parameters
    ----------
    count_rates : NDArray
        A 2D histogram array containing the
        count rates per spin at each energy bin.
    mean_duration : float
        Mean duration of the spins.
    sigma : int (default=6)
        The number of sigma.

    Returns
    -------
    threshold : NDArray
        Threshold for applying HIGHRATES flag.
    """
    sigma_per_energy = np.std(count_rates, axis=1)
    n_sigma_per_energy = sigma * sigma_per_energy
    mean_per_energy = np.mean(count_rates, axis=1)
    # Must have a HIGHRATES threshold of at least 3 counts per spin.
    threshold = np.maximum(mean_per_energy + n_sigma_per_energy, 3 / mean_duration)

    return threshold


def flag_spin(
    eventtimes_met: NDArray, energy: NDArray, sigma: int = 6
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
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
    spin : NDArray
        Spin data.
    energy_midpoints : NDArray
        Energy midpoint data.
    n_sigma_per_energy_reshape : NDArray
        N sigma per energy.
    """
    spin = get_spin(eventtimes_met)
    count_rates, spin_edges, counts, duration = get_energy_histogram(spin, energy)
    quality_flags = np.full(
        count_rates.shape, ImapRatesUltraFlags.NONE.value, dtype=np.uint16
    )

    # Zero counts/spin/energy level
    quality_flags[counts == 0] |= ImapRatesUltraFlags.ZEROCOUNTS.value
    threshold = get_n_sigma(count_rates, duration, sigma=sigma)

    bin_edges = np.array(UltraConstants.CULLING_ENERGY_BIN_EDGES)
    energy_midpoints = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    spin = np.unique(spin)

    # Indices where the counts exceed the threshold
    indices_n_sigma = count_rates > threshold[:, np.newaxis]
    quality_flags[indices_n_sigma] |= ImapRatesUltraFlags.HIGHRATES.value

    return quality_flags, spin, energy_midpoints, threshold
