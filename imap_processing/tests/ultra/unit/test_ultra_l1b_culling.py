"""Tests Culling for ULTRA L1b."""

import numpy as np

from imap_processing.quality_flags import ImapUltraFlags
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.ultra_l1b_culling import (
    flag_spin,
    get_energy_histogram,
    get_spin,
)


def test_get_spin(use_fake_spin_data_for_time, l1b_de_dataset):
    """Tests get_spin function."""
    use_fake_spin_data_for_time(
        l1b_de_dataset["event_times"][0], l1b_de_dataset["event_times"][-1]
    )
    spin_number, spin_start_time, spin_duration = get_spin(
        l1b_de_dataset["event_times"]
    )

    assert len(np.unique(spin_number)) == len(np.unique(spin_start_time))
    assert np.all(l1b_de_dataset["event_times"].values >= spin_start_time)


def test_get_energy_histogram(use_fake_spin_data_for_time, l1b_de_dataset):
    """Tests get_energy_histogram function."""

    use_fake_spin_data_for_time(
        l1b_de_dataset["event_times"][0], l1b_de_dataset["event_times"][-1]
    )

    spin_number, _, _ = get_spin(l1b_de_dataset["event_times"])
    hist, _ = get_energy_histogram(spin_number, l1b_de_dataset["energy"].values)

    assert hist.shape == (4, 15)

    energies_spin_0 = l1b_de_dataset["energy"].values[spin_number == 0]
    assert (
        len(
            energies_spin_0[
                energies_spin_0 >= UltraConstants.CULLING_ENERGY_BIN_EDGES[3]
            ]
        )
        == hist[3][0]
    )
    assert (
        len(
            energies_spin_0[
                (energies_spin_0 < UltraConstants.CULLING_ENERGY_BIN_EDGES[3])
                & (energies_spin_0 >= UltraConstants.CULLING_ENERGY_BIN_EDGES[2])
            ]
        )
        == hist[2][0]
    )

    energies_spin_14 = l1b_de_dataset["energy"].values[spin_number == 14]
    assert (
        len(
            energies_spin_14[
                energies_spin_14 >= UltraConstants.CULLING_ENERGY_BIN_EDGES[3]
            ]
        )
        == hist[3][14]
    )
    assert (
        len(
            energies_spin_14[
                (energies_spin_14 < UltraConstants.CULLING_ENERGY_BIN_EDGES[3])
                & (energies_spin_14 >= UltraConstants.CULLING_ENERGY_BIN_EDGES[2])
            ]
        )
        == hist[2][14]
    )


def test_flag_spin(use_fake_spin_data_for_time, l1b_de_dataset):
    """Tests flag_spin function."""

    use_fake_spin_data_for_time(
        l1b_de_dataset["event_times"][0], l1b_de_dataset["event_times"][-1]
    )

    spin_number, _, _ = get_spin(l1b_de_dataset["event_times"])
    energy = l1b_de_dataset["energy"].values
    hist, spin_edges = get_energy_histogram(spin_number, energy)

    quality_flags, spin, energy = flag_spin(l1b_de_dataset["event_times"], energy)

    flag = ImapUltraFlags(quality_flags[0])
    assert flag.name == "HIGHCOUNTS"

    flagged_indices = np.unique(spin)[hist[0, :] > 0]
    unflagged_indices = np.setdiff1d(np.unique(spin), flagged_indices)
    assert np.all(quality_flags[flagged_indices] == ImapUltraFlags.HIGHCOUNTS.value)
    assert np.all(quality_flags[unflagged_indices] == ImapUltraFlags.NONE.value)

    # Only HIGHCOUNT bits were set
    assert np.all(quality_flags == quality_flags & ImapUltraFlags.HIGHCOUNTS)
