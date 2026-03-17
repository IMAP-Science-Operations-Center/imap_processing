"""Tests Culling for ULTRA L1b."""

from unittest import mock

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
    flag_high_energy,
    flag_hk,
    flag_imap_instruments,
    flag_low_voltage,
    flag_rates,
    flag_scattering,
    flag_statistical_outliers,
    get_binned_energy_ranges,
    get_binned_spins_edges,
    get_de_rejection_mask,
    get_energy_and_spin_dependent_rejection_mask,
    get_energy_histogram,
    get_energy_range_flags,
    get_n_sigma,
    get_poisson_stats,
    get_pulses_per_spin,
    get_spin_data,
    get_valid_earth_angle_events,
    get_valid_events_per_energy_range,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import get_spin_info
from imap_processing.ultra.l1c.l1c_lookup_utils import build_energy_bins

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
    assert np.all(quality_flags)

    # Set only the first spin to be below threshold
    mock_status_dataset["rightdeflection_v"].data[1:] += 5000
    mock_status_dataset["leftdeflection_v"].data[1:] += 5000
    quality_flags = flag_low_voltage(spin_tbin_edges, mock_status_dataset)
    # Check that only the first spin is flagged for low voltage
    assert np.all(quality_flags[0])
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
    assert np.all(quality_flags)


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
    energy_range_flags = np.zeros(16, dtype=np.uint16)
    energy_range_flags[:3] = [2**1, 2**2, 2**3]  # Example flags for 3 energy bins
    energy_range_edges = np.full(17, -1.0e31, dtype=np.float32)
    energy_range_edges[:4] = [
        3,
        5,
        7,
        18,
    ]  # Example energy bin edges (4 edges = 3 bins)
    goodtimes_dataset = xr.Dataset(
        data_vars={
            "spin_number": np.arange(n_spins),
            "quality_low_voltage": np.full(n_spins, 0),
            "quality_high_energy": np.full(n_spins, 0),
            "quality_statistics": np.full(n_spins, 0),
            "energy_range_flags": energy_range_flags,
            "energy_range_edges": energy_range_edges,
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
    lv_threshold = 3000
    spin_tbin_edges = get_binned_spins_edges(
        xspin.spin_number.values,
        xspin.spin_period.values,
        xspin.spin_start_time.values,
        spin_bin_size,
    )
    lv_flags = flag_low_voltage(spin_tbin_edges, status_ds, lv_threshold)

    assert np.array_equal(lv_flags, validation_low_voltage_qf)


@mock.patch("imap_processing.ultra.l1b.ultra_l1b_culling.sp.spkezr")
def test_get_valid_earth_angle_events(mock_spkezr):
    """Tests get_valid_earth_angle_events function."""
    np.random.seed(0)
    de_dps_velocity = np.random.random((12, 3))
    de_dataset = xr.Dataset(
        {
            "velocity_dps_sc": (("epoch", "component"), de_dps_velocity),
            "event_times": ("epoch", np.arange(12)),
        }
    )
    earth_angle_threshold = np.radians(45)
    np.random.seed(0)
    mock_imap_state = np.random.random(6)  # Mock IMAP state for testing
    mock_spkezr.return_value = (mock_imap_state, None)
    # Calculate the expected flag exactly the way ULTRA IT does to ensure we are
    # getting the same results.
    # First negate the state vector: the state from imap_state(observer=EARTH)
    # gives the position of IMAP as seen from Earth (Earth to IMAP), while
    # the ULTRA code below expects the vector from IMAP to Earth.
    pos = mock_imap_state[:3]
    upos = pos / np.sqrt(np.sum(pos**2))
    zax0 = np.cross(upos, [0, 1, 0])
    zax = zax0 / np.sqrt(np.sum(zax0**2))
    yax = np.cross(zax, upos)

    vde = np.sqrt(np.sum(de_dps_velocity**2, 1))
    uv = de_dps_velocity / vde[:, np.newaxis]
    local_uv = np.array([upos, yax, zax]) @ np.transpose(-uv)
    expected_flags = local_uv[0, :] < np.cos(earth_angle_threshold)

    actual_flags = get_valid_earth_angle_events(de_dataset, earth_angle_threshold)
    np.testing.assert_array_equal(actual_flags, expected_flags)


def test_get_valid_events_per_energy_range():
    """Tests get_valid_events_per_energy_range function."""
    np.random.seed(0)
    energy_range_edges = np.array([3, 5, 7, 18])  # 3 example energy bins
    # example energy values that fall into different bins
    # - Events 1-3 and 8 fall into the second bin (5-7)
    # - Events 4-7 fall into the third bin (7-18)
    # - The rest of the event energies don't fall into any bins and should be
    # invalid for that energy range
    energy = np.array([5, 5, 6, 9, 10, 11, 12, 6, 7, 1, 20, 63])
    # Mark event 2 (energy bin 2) and 5 (energy bin 3) as outliers
    quality_outliers = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    # Mark event 1 (energy bin 2), 9 (energy bin 3), and 11 and 12 (No energy bin)
    quality_scattering = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1])
    ebin = np.full(len(energy), 10)
    # mark event 6 as having an invalid ebin
    ebin[5] = -1
    de_dps_velocity = np.random.random((len(energy), 3))

    de_dataset = xr.Dataset(
        {
            "de_dps_velocity": (("epoch", "component"), de_dps_velocity),
            "event_times": ("epoch", np.arange(len(energy))),
            "energy_spacecraft": ("epoch", energy),
            "quality_outliers": ("epoch", quality_outliers),
            "quality_scattering": ("epoch", quality_scattering),
            "ebin": ("epoch", ebin),
        }
    )
    keepout_angle = np.radians(180)
    valid_events = get_valid_events_per_energy_range(
        de_dataset, energy_range_edges, keepout_angle, 90
    )

    # Assert that for the first energy bin (3-5), all are false
    assert np.array_equal(valid_events[0], np.full(len(valid_events[0]), False))
    # Assert that for the second energy bin (5-7), all are false except
    # events 3 and 8 (event 1 had an outlier flag, event 2 had a scattering flag)
    expected_flags_ebin2 = np.array(
        [
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ]
    )
    assert np.array_equal(valid_events[1], expected_flags_ebin2)
    # Assert that for the third energy bin (7-18), all are false except events 4 and 7
    # (event 5 was marked as an outlier and event 6 has an invalid ebin)
    expected_flags_ebin3 = np.array(
        [
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
    )
    assert np.array_equal(valid_events[2], expected_flags_ebin3)


@mock.patch("imap_processing.ultra.l1b.ultra_l1b_culling.sp.spkezr")
def test_get_valid_events_per_energy_range_ultra45(mock_spkezr):
    """Tests get_valid_events_per_energy_range function."""
    np.random.seed(0)
    mock_imap_state = np.random.random(6)  # Mock IMAP state for testing
    mock_spkezr.return_value = (mock_imap_state, None)
    energy_range_edges = np.array([3, 5, 7, 18])  # 3 example energy bins
    energy = np.arange(18)

    # mark all events with valid outlier and scattering flags and valid ebins.
    de_dps_velocity = np.random.random((len(energy), 3))

    de_dataset = xr.Dataset(
        {
            "velocity_dps_sc": (("epoch", "component"), de_dps_velocity),
            "event_times": ("epoch", np.full(len(energy), 798033671)),
            "energy_spacecraft": ("epoch", energy),
            "quality_outliers": ("epoch", np.full(len(energy), 0)),
            "quality_scattering": ("epoch", np.full(len(energy), 0)),
            "ebin": ("epoch", np.full(len(energy), 10)),
        }
    )
    # ensure that all events fail the earth angle check by setting a very large
    # keepout angle
    keepout_angle = np.radians(360)
    valid_events = get_valid_events_per_energy_range(
        de_dataset, energy_range_edges, keepout_angle, 45
    )

    # although all events were valid for outliers, scattering, and ebin, all events
    # failed the earth angle check for ultra45
    assert not np.any(valid_events)


@mock.patch(
    "imap_processing.ultra.l1b.ultra_l1b_culling.UltraConstants.HIGH_ENERGY_CULL_CHANNEL",
    2,
)
@mock.patch(
    "imap_processing.ultra.l1b.ultra_l1b_culling.UltraConstants.HIGH_ENERGY_COMBINED_SPIN_BIN_RADIUS",
    None,
)
def test_flag_high_energy():
    """Tests flag_high_energy function."""
    # 7-18 is the culling energy bin and shown in the mock.patch above
    # Flag energy ranges will compare the counts from this energy range at each spin bin
    # to the culling threshold to determine if the spin bin should be flagged for high
    # energy.
    energy_range_edges = np.array([3, 5, 7, 18, 25])  # Example energy bin edges
    # Spin bin 1 (events 0-3) 4 events fall within the culling energy bin
    #   - This is above all the energy thresholds except the second one (10),
    #   so it should be flagged for all energy ranges except flag the 2nd
    # Spin bin 2 (events 4-7) only 1 event falls within the culling energy bin
    #   - This is above the lowest energy threshold (1) but below the rest, so
    #   it should only be flagged at the last energy range
    # Spin bin 3 (events 8-11) No events fall within the culling energy bin,
    #   so it should not be flagged for any energy range
    energy = np.array([17, 16, 12, 15, 5, 8, 4, 6, 4, 1, 22, 20])
    cull_thresholds = np.array([3, 10, 2, 1])
    de_dataset = xr.Dataset(
        {
            "de_event_met": ("epoch", np.arange(len(energy))),
            "energy_spacecraft": ("epoch", energy),
            "quality_outliers": ("epoch", np.full(len(energy), 0)),
            "quality_scattering": ("epoch", np.full(len(energy), 0)),
            "ebin": ("epoch", np.full(len(energy), 10)),
        }
    )
    # make one ebin invalid to make sure the valid events filtering is working
    de_dataset["ebin"].data[-1] = -1
    spin_tbin_edges = np.arange(
        start=0, stop=len(energy) + 1, step=4
    )  # create spin bins of 4 seconds
    quality_flags = flag_high_energy(
        de_dataset,
        spin_tbin_edges,
        energy_range_edges,
        None,
        cull_thresholds,
        90,
    )

    # check shape
    np.testing.assert_array_equal(
        quality_flags.shape, (len(energy_range_edges) - 1, len(spin_tbin_edges) - 1)
    )
    # Assert that the first spin bin is flagged for high energy for all energy ranges
    # except the second one
    assert quality_flags[0, 0]
    assert not quality_flags[1, 0]
    assert quality_flags[2, 0]
    assert quality_flags[3, 0]
    # Assert that the second spin bin is only flagged for high energy for the last
    # # energy range
    assert quality_flags[3, 1]
    assert not np.any(quality_flags[0:3, 1])
    # # Assert that the third spin bin is not flagged for any energy range
    assert not np.any(quality_flags[:, 2])


@mock.patch(
    "imap_processing.ultra.l1b.ultra_l1b_culling.UltraConstants.HIGH_ENERGY_CULL_CHANNEL",
    4,
)
@mock.patch(
    "imap_processing.ultra.l1b.ultra_l1b_culling.UltraConstants.HIGH_ENERGY_COMBINED_SPIN_BIN_RADIUS",
    3,
)
@pytest.mark.external_test_data
def test_validate_high_energy_cull():
    """Validate that high energy spins are correctly flagged"""
    # Mock thresholds to match the test data (I used fake ones to create more
    # complexity)
    mock_thresholds = np.array([0.05, 1.5, 0.6, 119.2, 0.2]) * 20
    # read test data from csv files
    xspin = pd.read_csv(TEST_PATH / "extendedspin_test_data_repoint00047.csv")
    expected_qf = pd.read_csv(
        TEST_PATH / "validate_high_energy_culling_results_repoint00047_v2.csv"
    ).to_numpy()
    de_df = pd.read_csv(TEST_PATH / "de_test_data_repoint00047.csv")
    de_ds = xr.Dataset(
        {
            "de_event_met": ("epoch", de_df.event_times.values),
            "energy_spacecraft": ("epoch", de_df.energy_spacecraft.values),
            "quality_outliers": ("epoch", de_df.quality_outliers.values),
            "quality_scattering": ("epoch", de_df.quality_scattering.values),
            "ebin": ("epoch", de_df.ebin.values),
        }
    )
    # Use constants from the code to ensure consistency with the actual culling code
    spin_bin_size = UltraConstants.SPIN_BIN_SIZE
    spin_tbin_edges = get_binned_spins_edges(
        xspin.spin_number.values,
        xspin.spin_period.values,
        xspin.spin_start_time.values,
        spin_bin_size,
    )
    # Get the energy ranges
    energy_ranges = np.array([4.2, 9.4425, 21.2116, 47.2388, 105.202, 316.335])
    e_flags = flag_high_energy(
        de_ds, spin_tbin_edges, energy_ranges, None, mock_thresholds
    )
    np.testing.assert_array_equal(e_flags, ~expected_qf.astype(bool))


def test_flag_statistical_outliers():
    """Tests flag_statistical_outliers function."""
    energy_range_edges = np.array([3, 5, 7, 18, 25])  # Example energy bin edges
    n_spin_bins = 12
    spin_step = 7
    energy = np.full(spin_step * n_spin_bins, 0)
    # Make sure there are at least 3 other bins with counts in each energy bin so that
    # the statistics can be calculated.
    energy[::spin_step] = 3
    energy[1::spin_step] = 5
    energy[2::spin_step] = 7
    energy[3::spin_step] = 18
    # Make the last spin bin have higher counts. It should get flagged as an outlier.
    energy[-spin_step:] = 23

    de_dataset = xr.Dataset(
        {
            "de_event_met": ("epoch", np.arange(len(energy))),
            "energy_spacecraft": ("epoch", energy),
            "quality_outliers": ("epoch", np.full(len(energy), 0)),
            "quality_scattering": ("epoch", np.full(len(energy), 0)),
            "ebin": ("epoch", np.full(len(energy), 10)),
        }
    )
    spin_tbin_edges = np.arange(
        start=0, stop=len(energy) + 1, step=spin_step
    )  # create spin bins of 7 seconds
    quality_flags, convergence, iterations, std_diff = flag_statistical_outliers(
        de_dataset,
        spin_tbin_edges,
        energy_range_edges,
        np.zeros((len(energy_range_edges) - 1, len(spin_tbin_edges) - 1), dtype=bool),
        combine_flags_across_energy_bins=True,
    )

    # check shape
    np.testing.assert_array_equal(
        quality_flags.shape, (len(energy_range_edges) - 1, len(spin_tbin_edges) - 1)
    )
    # check that none of the flags are set except for the last spin bin
    # since combine_flags_across_energy_bins is True, the entire last spin bin should
    # be flagged even though only one energy bin had high counts
    expected_flags = np.zeros(
        (len(energy_range_edges) - 1, len(spin_tbin_edges) - 1), dtype=bool
    )
    expected_flags[:, -1] = True
    np.testing.assert_array_equal(quality_flags, expected_flags)
    # all energy bins should have converged
    # The first 2 didn't have enough events to calculate statistics, but they should
    # still be marked as converged
    assert np.all(convergence)
    # All energy bins should have iterated 1 time except the last one which should have
    # iterated twice.
    assert np.all(iterations[:-1] == 1)
    assert iterations[-1] == 2
    # Check that all std_diff values were set (not zero) except the last one
    assert np.all(std_diff[:-1] != 0)
    assert std_diff[-1] == 0


def test_flag_statistical_outliers_invalid_events():
    """Tests flag_statistical_outliers function when there are no valid events."""
    energy_range_edges = np.array([3, 5, 7, 18, 25])  # Example energy bin edges
    energy = np.arange(25)
    de_dataset = xr.Dataset(
        {
            "de_event_met": ("epoch", np.arange(len(energy))),
            "energy_spacecraft": ("epoch", energy),
            "quality_outliers": ("epoch", np.full(len(energy), 0)),
            "quality_scattering": ("epoch", np.full(len(energy), 0)),
            "ebin": ("epoch", np.full(len(energy), 10)),
        }
    )
    spin_tbin_edges = np.arange(
        start=0, stop=len(energy) + 1, step=5
    )  # create spin bins of 5 seconds
    mask = np.ones((len(energy_range_edges) - 1, len(spin_tbin_edges) - 1), dtype=bool)
    quality_flags, convergence, iterations, std_diff = flag_statistical_outliers(
        de_dataset,
        spin_tbin_edges,
        energy_range_edges,
        mask,
    )
    # check that no flags are set because there were no valid events to calculate
    # statistics on.
    np.testing.assert_array_equal(
        quality_flags, np.zeros_like(quality_flags, dtype=bool)
    )
    # check that all energy bins are marked as converged (no valid events is not a
    # failure case for convergence since we just can't calculate statistics.
    assert np.all(convergence)
    # check that there were no iterations
    assert np.sum(iterations) == 0
    # Check that std_diff is all invalid (-1)
    assert np.all(std_diff == -1)


def test_get_poisson_stats():
    """Tests get_poisson_stats function."""
    counts = np.full(20, 0)
    counts[-1] = 100  # Make the last bin have high counts
    std_diff, outlier_mask = get_poisson_stats(counts)
    # std_diff should be counts/sqrt(counts) = sqrt(counts)
    assert (
        std_diff == np.std(counts) / np.sqrt(5) - 1
    )  # std_diff should be counts/sqrt(counts) = sqrt(counts)
    assert (
        np.sum(outlier_mask) == 1
    )  # Only the last bin should be flagged as an outlier
    assert outlier_mask[-1]


@pytest.mark.external_test_data
def test_validate_stat_cull():
    """Validate that statistical-outlier quality flags match expected results."""
    # read test data from csv files
    xspin = pd.read_csv(TEST_PATH / "extendedspin_test_data_repoint00047.csv")
    results_df = pd.read_csv(
        TEST_PATH / "validate_stat_culling_results_repoint00047_v2.csv"
    )
    de_df = pd.read_csv(TEST_PATH / "de_test_data_repoint00047.csv")
    de_ds = xr.Dataset(
        {
            "de_event_met": ("epoch", de_df.event_times.values),
            "energy_spacecraft": ("epoch", de_df.energy_spacecraft.values),
            "quality_outliers": ("epoch", de_df.quality_outliers.values),
            "quality_scattering": ("epoch", de_df.quality_scattering.values),
            "ebin": ("epoch", de_df.ebin.values),
        }
    )
    # Use constants from the code to ensure consistency with the actual culling code
    spin_bin_size = UltraConstants.SPIN_BIN_SIZE
    spin_tbin_edges = get_binned_spins_edges(
        xspin.spin_number.values,
        xspin.spin_period.values,
        xspin.spin_start_time.values,
        spin_bin_size,
    )
    # Get the energy ranges
    energy_ranges = np.array([4.2, 9.4425, 21.2116, 47.2388, 105.202, 316.335])

    # Create a mask of flagged events to test that the stat cull algorithm
    # properly ignores these. The test data was created using this exact mask as well.
    mask = np.zeros((len(energy_ranges) - 1, len(spin_tbin_edges) - 1), dtype=bool)
    mask[0:2, 0:2] = (
        True  # This will mark the first 2 energy bins and first 2 spin bins as flagged
    )
    # ignored in the statistics calculation and flagging.
    flags, con, it, std = flag_statistical_outliers(
        de_ds, spin_tbin_edges, energy_ranges, mask, 90
    )
    expected_qf = results_df.iloc[:, :-3].values.astype(bool)
    converge = results_df["converge"].values
    iterations = results_df["iterations"].values
    std_diff = results_df["std_diff"].values
    # check that the flags match the expected results
    np.testing.assert_array_equal(flags, ~expected_qf.astype(bool))
    np.testing.assert_array_equal(con, converge)
    np.testing.assert_array_equal(it, iterations)
    np.testing.assert_allclose(std, std_diff, rtol=1e-10)


def test_get_energy_range_flags():
    """Tests get_binned_energy_range_flags function."""
    # Get energy bins used at l1c
    intervals, _, _ = build_energy_bins()
    # Get the energy ranges
    energy_ranges = get_binned_energy_ranges(intervals)
    flags = get_energy_range_flags(energy_ranges)

    np.testing.assert_array_equal(flags, 2 ** np.arange(6))


def test_get_binned_energy_ranges():
    """Tests get_binned_energy_ranges function."""
    intervals, _, _ = build_energy_bins()
    energy_ranges = get_binned_energy_ranges(intervals)
    expected_energy_ranges = np.array(
        [3.0, 6.96, 15.71, 34.9866, 77.9161, 116.276, 316.335]
    )
    np.testing.assert_array_equal(energy_ranges, expected_energy_ranges)
