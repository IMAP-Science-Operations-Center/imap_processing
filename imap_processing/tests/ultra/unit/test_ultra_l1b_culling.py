"""Tests Culling for ULTRA L1b."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.quality_flags import (
    ImapAttitudeUltraFlags,
    ImapDEScatteringUltraFlags,
    ImapHkUltraFlags,
    ImapInstrumentUltraFlags,
    ImapRatesUltraFlags,
)
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.ultra_l1b_culling import (
    compare_aux_univ_spin_table,
    count_rejected_events_per_spin,
    expand_bin_flags_to_spins,
    flag_attitude,
    flag_hk,
    flag_imap_instruments,
    flag_low_voltage,
    flag_rates,
    flag_scattering,
    get_binned_spins_edges,
    get_de_rejection_mask,
    get_energy_and_spin_dependent_rejection_mask,
    get_energy_histogram,
    get_n_sigma,
    get_pulses_per_spin,
    get_spin_data,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import get_spin_info

TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.fixture
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
        faux_aux_dataset["spinnumber"].values, faux_aux_dataset
    )

    flag = ImapAttitudeUltraFlags(quality_flags[0])
    assert flag.name == "NONE"
    assert quality_flags[-1] == ImapAttitudeUltraFlags.AUXMISMATCH.value
    assert np.all(spin_rates == 60 / spin_period)
    assert np.all(np.diff(spin_start_time) == 15)

    spins = np.unique(faux_aux_dataset["spinnumber"].values)  # Get unique spins
    spin_df = get_spin_data()
    spin_phase_valid = spin_df.loc[spin_df.spin_number.isin(spins), "spin_phase_valid"]
    spin_period_valid = spin_df.loc[
        spin_df.spin_number.isin(spins), "spin_period_valid"
    ]

    assert np.all(
        quality_flags[spin_phase_valid == 0] & ImapAttitudeUltraFlags.SPINPHASE.value
    )
    assert np.all(
        quality_flags[~spin_period_valid] & ImapAttitudeUltraFlags.SPINPERIOD.value
    )


def test_get_n_sigma():
    """Tests get_six_sigma function."""

    counts = np.array([[16, 4, 1], [0, 0, 0], [1, 1, 1], [2, 0, 5]])
    threshold = get_n_sigma(counts / 15, 15, 6)

    assert np.all(threshold >= 3 / 15)
    mean = np.mean(counts[0] / 15)
    squared_differences = (counts[0] / 15 - mean) ** 2
    variance = np.sum(squared_differences) / (counts.shape[1] - 1)
    std_dev = np.sqrt(variance)

    np.testing.assert_allclose(mean + std_dev * 6, threshold[0], atol=1e-2, rtol=0)


def test_flag_hk(test_data):
    """Tests flag_hk function."""

    spin_number, _, _ = test_data
    hk_qf = flag_hk(spin_number)

    assert np.all(hk_qf == ImapHkUltraFlags.NONE.value)


def test_flag_imap_instruments(test_data):
    """Tests flag_imap_instruments function."""

    spin_number, _, _ = test_data
    hk_qf = flag_imap_instruments(spin_number)

    assert np.all(hk_qf == ImapInstrumentUltraFlags.NONE.value)


def test_flag_rates(test_data):
    """Tests flag_rates function."""

    spin_number, energy, expected_counts = test_data
    quality_flags, spin, energy, _ = flag_rates(spin_number, energy, 1)
    threshold = get_n_sigma(expected_counts / 15, 15, 1)

    expected_quality_flags = np.full(
        (len(UltraConstants.CULLING_ENERGY_BIN_EDGES) - 1, len(np.unique(spin))),
        ImapRatesUltraFlags.NONE.value,
        dtype=np.uint16,
    )
    expected_quality_flags[:, 0] |= ImapRatesUltraFlags.FIRSTSPIN.value
    expected_quality_flags[:, -1] |= ImapRatesUltraFlags.LASTSPIN.value

    assert np.array_equal(
        quality_flags[expected_counts == 0],
        expected_quality_flags[expected_counts == 0],
    )
    high_rates_flag = quality_flags[expected_counts / 15 > threshold[:, np.newaxis]]
    assert np.all(
        high_rates_flag
        == ImapRatesUltraFlags.HIGHRATES.value | ImapRatesUltraFlags.FIRSTSPIN.value
    )


def test_compare_aux_univ_spin_table(use_fake_spin_data_for_time, faux_aux_dataset):
    """Tests compare_aux_univ_spin_table function."""
    use_fake_spin_data_for_time(0, 15 * 147)
    spins = faux_aux_dataset["spinnumber"].values
    spin_df = get_spin_data()

    result = compare_aux_univ_spin_table(faux_aux_dataset, spins, spin_df)
    expected = np.array([False] * 14 + [True])

    assert np.all(result == expected)


def test_get_duration(rates_l1_test_path, use_fake_spin_data_for_time):
    """Tests get_duration function."""
    use_fake_spin_data_for_time(start_met=0, end_met=141 * 15)
    df = pd.read_csv(rates_l1_test_path)

    # Should be evenly spaced spins of 15 seconds each except the first one has 14.
    num_spins = 15
    spin_start_times = np.concatenate([[0], np.arange(14, 222, num_spins)])
    spin_numbers = np.arange(127, 142)
    num_spins = len(spin_numbers)

    aux_ds = xr.Dataset(
        data_vars={
            "timespinstart": ("epoch", spin_start_times),
            "timespinstartsub": ("epoch", np.ones_like(spin_start_times)),
            "duration": ("epoch", np.full(num_spins, 15)),
            "spinnumber": ("epoch", spin_numbers),
        },
        coords={"epoch": ("epoch", np.arange(num_spins))},
    )

    met = df["TimeTag"] - df["TimeTag"].values[0]
    spin = df["Spin"]
    spin_ds = get_spin_info(aux_ds, met)
    spin_number = spin_ds["spin_number"].values
    duration = spin_ds["spin_duration"].values
    assert np.array_equal(spin, spin_number)
    assert np.all(duration == 15)


def test_get_pulses(rates_l1_test_path, use_fake_spin_data_for_time, aux_dataset):
    """Tests get_pulses_per_spin function."""
    df = pd.read_csv(rates_l1_test_path)

    # Simulate a spin table from MET = 0 to MET = 141 * 15 seconds
    use_fake_spin_data_for_time(start_met=0, end_met=141 * 15)

    pulse_dict = {
        # Stop pulses
        "stop_tn": df["StopTopNorthCFD"],
        "stop_bn": df["StopBottomNorthCFD"],
        "stop_te": df["StopTopEastCFD"],
        "stop_be": df["StopBottomEastCFD"],
        "stop_ts": df["StopTopSouthCFD"],
        "stop_bs": df["StopBottomSouthCFD"],
        "stop_tw": df["StopTopWestCFD"],
        "stop_bw": df["StopBottomWestCFD"],
        # Start pulses
        "start_rf": df["StartRightFullCFD"],
        "start_lf": df["StartLeftFullCFD"],
        # Coincidence pulses
        "coin_tn": df["CoinTopNorthCFD"],
        "coin_bn": df["CoinBottomNorthCFD"],
        "coin_ts": df["CoinTopSouthCFD"],
        "coin_bs": df["CoinBottomSouthCFD"],
        # Additional info
        "shcoarse": df["TimeTag"],
        "spin": df["Spin"],
    }

    pulses = get_pulses_per_spin(aux_dataset, pulse_dict)
    unique_spins = np.unique(pulse_dict["spin"])

    start_pulses_total = pulse_dict["start_rf"] + pulse_dict["start_lf"]
    stop_pulses_total = np.max(
        np.stack([v for k, v in pulse_dict.items() if k.startswith("stop_t")], axis=1),
        axis=1,
    ) + np.max(
        np.stack([v for k, v in pulse_dict.items() if k.startswith("stop_b")], axis=1),
        axis=1,
    )
    coin_pulses_total = np.max(
        np.stack([v for k, v in pulse_dict.items() if k.startswith("coin_t")], axis=1),
        axis=1,
    ) + np.max(
        np.stack([v for k, v in pulse_dict.items() if k.startswith("coin_b")], axis=1),
        axis=1,
    )

    for i, spin in enumerate(unique_spins):
        mask = pulse_dict["spin"] == spin
        assert np.isclose(pulses.start_per_spin[i], np.sum(start_pulses_total[mask]))
        assert np.isclose(pulses.stop_per_spin[i], np.sum(stop_pulses_total[mask]))
        assert np.isclose(pulses.coin_per_spin[i], np.sum(coin_pulses_total[mask]))

    np.testing.assert_allclose(pulses.start_pulses, start_pulses_total)
    np.testing.assert_allclose(pulses.stop_pulses, stop_pulses_total)
    np.testing.assert_allclose(pulses.coin_pulses, coin_pulses_total)


@pytest.mark.external_test_data
def test_flag_scattering(ancillary_files):
    """Tests flag_scattering function."""
    tof_energy = np.full(9, 0.5)
    theta = np.full(9, 30.0)
    phi = np.full(9, 60.0)
    quality_flags = np.full(
        phi.shape, ImapDEScatteringUltraFlags.NONE.value, dtype=np.uint16
    )
    flag_scattering(tof_energy, theta, phi, ancillary_files, "ultra45", quality_flags)
    assert np.all(quality_flags == 0)

    tof_energy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    theta = np.array([1, 2, 50, 50, 50, 60, 70, 80, 90])
    phi = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    quality_flags = np.full(
        phi.shape, ImapDEScatteringUltraFlags.NONE.value, dtype=np.uint16
    )
    flag_scattering(tof_energy, theta, phi, ancillary_files, "ultra45", quality_flags)
    assert np.all(quality_flags == np.array([1, 1, 2, 2, 2, 2, 2, 2, 2]))


def test_get_de_rejection_mask():
    """Tests get_de_rejection_mask function."""
    quality_scattering = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0])
    quality_outliers = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0])

    counted = get_de_rejection_mask(quality_scattering, quality_outliers)

    np.testing.assert_array_equal(
        counted, np.array([False, True, True, True, True, False, True, True, False])
    )

    counts_no_scattering = get_de_rejection_mask(
        quality_scattering, quality_outliers, reject_scattering=False
    )
    np.testing.assert_array_equal(counts_no_scattering, quality_outliers.astype(bool))


def test_count_rejected_events_per_spin():
    """Tests count_rejected_events_per_spin function."""

    spins = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    quality_scattering = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0])
    quality_outliers = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0])

    counted = count_rejected_events_per_spin(
        spins, quality_scattering, quality_outliers
    )

    np.testing.assert_array_equal(counted, np.array([2, 2, 2]))


def test_flag_low_voltage(test_data):
    """Tests flag_low_voltage function."""
    n_spins = 20
    mock_status_dataset = xr.Dataset(
        data_vars={
            "shcoarse": np.arange(n_spins),
            # Set Voltage below threshold
            "rightdeflection_v": np.full(n_spins, 0.5),
            "leftdeflection_v": np.full(n_spins, 1.5),
        }
    )
    flagged = 65535
    spins = np.arange(n_spins)
    spin_bin_size = 5
    spin_period = np.full(n_spins, 15.0)
    spin_starttime = np.arange(n_spins)
    spin_tbin_edges = get_binned_spins_edges(
        spins, spin_period, spin_starttime, spin_bin_size
    )
    quality_flags = flag_low_voltage(spin_tbin_edges, mock_status_dataset)

    # There should be an extra bin edge for the last bin to indicate the end of the last
    # spin bin
    assert len(spin_tbin_edges) == (n_spins // 5) + 1
    # Check quality flag shape
    assert quality_flags.shape == (len(spin_tbin_edges) - 1,)
    # Check that every spin is flagged for low voltage
    assert np.all(quality_flags == flagged)

    # Set only the first spin to be below threshold
    mock_status_dataset["rightdeflection_v"].data[1:] += 5000
    mock_status_dataset["leftdeflection_v"].data[1:] += 5000
    quality_flags = flag_low_voltage(spin_tbin_edges, mock_status_dataset)
    # Check that only the first spin is flagged for low voltage
    assert np.all(quality_flags[0] == flagged)
    # The rest should not be flagged
    assert np.all(quality_flags[1:] == 0)


def test_flag_low_voltage_incomplete_bins(test_data):
    """Tests flag_low_voltage function when there is an incomplete spin bin."""
    n_spins = 12  # Not a multiple of spin_bin_size to test incomplete bins
    mock_status_dataset = xr.Dataset(
        data_vars={
            "shcoarse": np.arange(n_spins),
            # Set Voltage below threshold
            "rightdeflection_v": np.full(n_spins, 0.5),
            "leftdeflection_v": np.full(n_spins, 1.5),
        }
    )

    spins = np.arange(n_spins)
    spin_bin_size = 5
    spin_period = np.full(n_spins, 15.0)
    spin_starttime = np.arange(n_spins)
    spin_tbin_edges = get_binned_spins_edges(
        spins, spin_period, spin_starttime, spin_bin_size
    )
    quality_flags = flag_low_voltage(spin_tbin_edges, mock_status_dataset)

    # check quality flag
    assert quality_flags.shape == (n_spins // spin_bin_size,)
    # Check that every spin is flagged for low voltage
    flagged = 65535
    assert np.all(quality_flags == flagged)


def test_expand_bin_flags_to_spins(caplog):
    """Tests expand_bin_flags_to_spins function."""
    spin_bin_size = 5
    n_spins = 12
    # Mock the shape of binned quality flags for 12 spins and a bin size of 5
    binned_qf = np.full((n_spins // spin_bin_size), 1)
    quality_flags = expand_bin_flags_to_spins(n_spins, binned_qf, spin_bin_size)
    # Check the size
    assert quality_flags.shape == (n_spins,)
    # The first 10 spins should be flagged since they fall into the first two bins
    assert np.all(quality_flags[:10] == 1)
    # The last 2 spins should not be flagged since they fall into the last incomplete
    # bin
    assert np.all(quality_flags[10:] == 0)
    binned_qf = np.full((n_spins // spin_bin_size) + 1, 1)
    # test that a warning is logged when there are incomplete bins found
    expand_bin_flags_to_spins(n_spins, binned_qf, spin_bin_size)
    assert "Found incomplete spin bin at the end with 3 spins" in caplog.text


def test_get_energy_and_spin_dependent_rejection_mask():
    """Tests get_energy_and_spin_dependent_rejection_mask function."""
    n_spins = 10
    goodtimes_dataset = xr.Dataset(
        data_vars={
            "spin_number": np.arange(n_spins),
            "quality_low_voltage": np.full(n_spins, 0),
            "quality_high_energy": np.full(n_spins, 0),
            "quality_statistics": np.full(n_spins, 0),
            "energy_range_flags": np.array(
                [2**1, 2**2, 2**3]
            ),  # Example flags for energy bins
            "energy_range_edges": np.array([3, 5, 7, 18]),  # Example energy bin edges
        }
    )
    # update quality flags to test that events get rejected
    # For spin 0, set energy bin 0 to be bad (flag = 2)
    goodtimes_dataset["quality_low_voltage"].data[0] = 2
    # For spin 2, set energy bin 1 to be bad (flag = 4)
    goodtimes_dataset["quality_high_energy"].data[2] = 4
    # For spin 4, set energy bin 2 to be bad (flag = 8)
    # Energy corresponding to spin 5 will not be rejected since it is not
    # within an energy bin
    goodtimes_dataset["quality_high_energy"].data[4] = 8
    # Create 6 fake events
    energy = np.array(
        [4, 5, 6, 9, 18, 15]
    )  # Energy values that fall into different bins
    spin_number = np.arange(6)
    rejected = get_energy_and_spin_dependent_rejection_mask(
        goodtimes_dataset, energy, spin_number
    )

    np.testing.assert_array_equal(
        rejected, np.array([True, False, True, False, False, False])
    )


@pytest.mark.external_test_data
def test_validate_voltage_cull():
    """Validate that low voltage spins are correctly flagged"""
    # read test data from csv files
    xspin = pd.read_csv(TEST_PATH / "extendedspin_test_data_repoint00047.csv")
    validation_low_voltage_qf = np.loadtxt(
        TEST_PATH / "voltage_culling_results_repoint00047.csv",
        delimiter=",",
        dtype=np.uint16,
    )
    status_df = pd.read_csv(TEST_PATH / "status_test_data_repoint00047.csv")
    # build the status dataset including the variables needed for the low voltage flag
    status_ds = xr.Dataset(
        {
            "shcoarse": ("epoch", status_df.shcoarse.values),
            "rightdeflection_v": ("epoch", status_df.rightdeflection_v.values),
            "leftdeflection_v": ("epoch", status_df.leftdeflection_v.values),
        }
    )
    # Use constants from the code to ensure consistency with the actual culling code
    spin_bin_size = UltraConstants.SPIN_BIN_SIZE
    lv_threshold = UltraConstants.LOW_VOLTAGE_CULL_THRESHOLD
    spin_tbin_edges = get_binned_spins_edges(
        xspin.spin_number.values,
        xspin.spin_period.values,
        xspin.spin_start_time.values,
        spin_bin_size,
    )
    lv_flags = flag_low_voltage(
        spin_tbin_edges, status_ds, lv_threshold, low_voltage_flag=1
    )

    assert np.array_equal(lv_flags, validation_low_voltage_qf)
