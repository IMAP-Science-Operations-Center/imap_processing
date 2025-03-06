"""Tests Culling for ULTRA L1b."""

import numpy as np
import pytest

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.ultra_l1b_culling import (
    compare_aux_univ_spin_table,
    flag_attitude,
    flag_spin,
    get_energy_histogram,
    get_n_sigma,
    get_spin_data,
)


@pytest.fixture()
def test_data(use_fake_spin_data_for_time):
    """Fixture to compute and return test data."""

    time = np.arange(0, 32, 2)
    spin_number = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2])
    energy = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 15, 15, 25, -2, -2, -2, 2])

    use_fake_spin_data_for_time(time[0], time[-1])

    energy_edges = UltraConstants.CULLING_ENERGY_BIN_EDGES
    unique_spins = np.unique(spin_number)
    expected_counts = np.zeros((len(energy_edges) - 1, len(unique_spins)))

    for spin_idx, spin in enumerate(unique_spins):
        for energy_idx in range(len(energy_edges) - 1):
            count = np.sum(
                (spin_number == spin)
                & (energy >= energy_edges[energy_idx])
                & (energy < energy_edges[energy_idx + 1])
            )
            expected_counts[energy_idx, spin_idx] = count

    return spin_number, energy, expected_counts


def test_get_energy_histogram(test_data):
    """Tests get_energy_histogram function."""

    spin_number, energy, expected_counts = test_data

    hist, _, counts, duration = get_energy_histogram(spin_number, energy)

    assert np.all(counts == expected_counts)
    assert np.all(hist == expected_counts / 15)
    assert duration == 15


def test_flag_attitude(use_fake_spin_data_for_time, faux_aux_dataset):
    """Tests flag_attitude function."""

    use_fake_spin_data_for_time(0, 15 * 147)
    quality_flags, spin_rates, spin_period, spin_start_time = flag_attitude(
        faux_aux_dataset["SPINNUMBER"].values, faux_aux_dataset
    )

    flag = ImapAttitudeUltraFlags(quality_flags[0])
    assert flag.name == "NONE"
    assert quality_flags[-1] == ImapAttitudeUltraFlags.AUXMISMATCH.value
    assert np.all(spin_rates == 60 / spin_period)
    assert np.all(np.diff(spin_start_time) == 15)


def test_get_n_sigma():
    """Tests get_six_sigma function."""

    counts = np.array([[16, 4, 1], [0, 0, 0], [1, 1, 1], [2, 0, 5]])
    threshold = get_n_sigma(counts / 15, 15, 6)

    assert np.all(threshold >= 3 / 15)
    mean = np.mean(counts[0] / 15)
    squared_differences = (counts[0] / 15 - mean) ** 2
    variance = np.mean(squared_differences)
    std_dev = np.sqrt(variance)

    np.testing.assert_allclose(mean + std_dev * 6, threshold[0], atol=1e-2, rtol=0)


def test_flag_spin(test_data):
    """Tests flag_spin function."""

    spin_number, energy, expected_counts = test_data
    quality_flags, spin, energy, _ = flag_spin(spin_number, energy, 1)
    threshold = get_n_sigma(expected_counts / 15, 15, 1)

    # At the first energy level were the rates > threshold and the counts > threshold?
    assert np.all(
        quality_flags[expected_counts == 0] == ImapRatesUltraFlags.ZEROCOUNTS.value
    )
    high_rates_flag = quality_flags[expected_counts / 15 > threshold[:, np.newaxis]]
    assert np.all(high_rates_flag == ImapRatesUltraFlags.HIGHRATES.value)


def test_compare_aux_univ_spin_table(use_fake_spin_data_for_time, faux_aux_dataset):
    """Tests compare_aux_univ_spin_table function."""
    use_fake_spin_data_for_time(0, 15 * 147)
    spins = faux_aux_dataset["SPINNUMBER"].values
    spin_df = get_spin_data()

    result = compare_aux_univ_spin_table(faux_aux_dataset, spins, spin_df)
    expected = np.array([False] * 14 + [True])

    assert np.all(result == expected)
