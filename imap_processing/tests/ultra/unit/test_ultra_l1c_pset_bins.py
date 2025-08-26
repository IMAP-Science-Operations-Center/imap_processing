"Tests pointing sets"

from unittest import mock

import astropy_healpix.healpy as hp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ultra.l1c import ultra_l1c_pset_bins
from imap_processing.ultra.l1c.spacecraft_pset import (
    calculate_pixels_within_scattering_threshold,
)
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    apply_deadtime_correction,
    build_energy_bins,
    get_deadtime_ratios,
    get_deadtime_ratios_by_spin_phase,
    get_energy_delta_minus_plus,
    get_helio_adjusted_data,
    get_sectored_rates,
    get_spacecraft_background_rates,
    get_spacecraft_count_rate_uncertainty,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
)

BASE_PATH = imap_module_directory / "ultra" / "lookup_tables"
TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.fixture
def test_data():
    """Test data fixture."""
    vx_sc = np.array([-186.5575, 508.5697, 508.5697, 508.5697])
    vy_sc = np.array([-707.5707, -516.0282, -516.0282, -516.0282])
    vz_sc = np.array([618.0569, 892.6931, 892.6931, 892.6931])
    energy = np.array([3.384, 3.385, 4.138, 4.138])
    v = np.column_stack((vx_sc, vy_sc, vz_sc))

    return v, energy


def test_build_energy_bins():
    """Tests build_energy_bins function."""
    intervals, energy_midpoints, energy_bin_geometric_means = build_energy_bins()
    energy_bin_start = [interval[0] for interval in intervals]
    energy_bin_end = [interval[1] for interval in intervals]

    assert energy_bin_start[0] == 3.385
    assert np.allclose(energy_bin_start[1], 4.137, atol=1e-3)
    assert len(intervals) == 24
    assert energy_midpoints[0] == (energy_bin_start[0] + energy_bin_end[0]) / 2

    # Comparison to expected values.
    np.testing.assert_allclose(energy_bin_end[1], 5.056, atol=1e-3)
    np.testing.assert_allclose(energy_bin_start[-1], 341.989, atol=1e-3)
    np.testing.assert_allclose(energy_bin_end[-1], 100000, atol=1e-3)

    expected_geometric_means = np.sqrt(
        np.array(energy_bin_start) * np.array(energy_bin_end)
    )
    np.testing.assert_allclose(
        energy_bin_geometric_means, expected_geometric_means, atol=1e-4
    )


def test_get_energy_delta_minus_plus(monkeypatch):
    """Tests get_energy_delta_minus_plus function."""
    # Mock fixed values for the energy bins - these are not the actual geometric means
    mock_intervals = [(0, 1), (1, 5), (5, 20), (20, 1234)]
    mock_midpoints = None
    mock_geometric_means = np.array([0, 2, 7, 100])

    expected_bins_energy_delta_plus = np.array([1, 3, 13, 1134])
    expected_bins_energy_delta_minus = np.array([0, 1, 2, 80])

    def mock_build_energy_bins():
        return mock_intervals, mock_midpoints, mock_geometric_means

    monkeypatch.setattr(
        ultra_l1c_pset_bins, "build_energy_bins", mock_build_energy_bins
    )

    bins_energy_delta_minus, bins_energy_delta_plus = get_energy_delta_minus_plus()
    assert np.array_equal(bins_energy_delta_plus, expected_bins_energy_delta_plus)
    assert np.array_equal(bins_energy_delta_minus, expected_bins_energy_delta_minus)


def test_get_spacecraft_histogram(test_data):
    """Tests get_histogram function."""
    v, energy = test_data

    energy_bin_edges, _, _ = build_energy_bins()
    subset_energy_bin_edges = energy_bin_edges[:3]

    hist, latitude, longitude, n_pix = get_spacecraft_histogram(
        v, energy, subset_energy_bin_edges, nside=1
    )
    assert hist.shape == (len(subset_energy_bin_edges), hp.nside2npix(1))
    assert n_pix == hp.nside2npix(1)
    assert latitude.shape == (n_pix,)
    assert longitude.shape == (n_pix,)

    # Spot check that 1 count is in the first energy bin
    assert np.sum(hist[1, :]) == 2

    # Test overlapping energy bins
    overlapping_bins = [
        (0.0, 3.385),
        (2.5, 4.137),
        (3.385, 5.057),
    ]
    hist, latitude, longitude, n_pix = get_spacecraft_histogram(
        v, energy, overlapping_bins, nside=1
    )
    # Spot check that 3 counts are in the third energy bin
    assert np.sum(hist[2, :]) == 3
    assert n_pix == hp.nside2npix(1)
    assert latitude.shape == (n_pix,)
    assert longitude.shape == (n_pix,)


def mock_imap_state(time, ref_frame):
    # Position (0, 0, 0), exaggerated velocity to force visible transformation
    return np.array([0, 0, 0, 0, 0, 0])


def test_get_sectored_rates():
    """Tests get_sectored_rates function."""

    # Simulate a test rates dataset.
    epoch = 60
    test_l1a_rates_dataset = xr.Dataset(
        {
            "test_data": (["epoch"], np.arange(epoch)),
        },
    )
    # Sector mode (image rates cadence = 3) happens 3 times a day (per pointing).
    # each time the mode changes, it is recorded in the params packet.
    # Create a test params dataset that simulates the mode changing to 3, 3 times.
    modes = np.tile(np.array([1, 3]), 3)
    test_l1a_params_dataset = xr.Dataset(
        {
            "imageratescadence": (["epoch"], modes),
        },
        coords={"epoch": ("epoch", np.arange(0, epoch, epoch / len(modes)))},
    )
    sectored_rates = get_sectored_rates(test_l1a_rates_dataset, test_l1a_params_dataset)
    np.testing.assert_array_equal(
        sectored_rates["test_data"].data,
        np.hstack([np.arange(10, 20), np.arange(30, 40), np.arange(50, 60)]),
    )
    # Test with one mode shift in the middle of the dataset.
    modes = np.array([1, 3, 1])
    test_l1a_params_dataset = xr.Dataset(
        {
            "imageratescadence": (["epoch"], modes),
        },
        coords={"epoch": ("epoch", np.arange(0, epoch, epoch / len(modes)))},
    )
    sectored_rates = get_sectored_rates(test_l1a_rates_dataset, test_l1a_params_dataset)
    np.testing.assert_array_equal(sectored_rates["test_data"].data, np.arange(20, 40))

    # Test with one mode shift in the middle of the dataset.
    modes = np.array([1, 3, 1])
    test_l1a_params_dataset = xr.Dataset(
        {
            "imageratescadence": (["epoch"], modes),
        },
        coords={"epoch": ("epoch", np.arange(0, epoch, epoch / len(modes)))},
    )
    sectored_rates = get_sectored_rates(test_l1a_rates_dataset, test_l1a_params_dataset)
    np.testing.assert_array_equal(sectored_rates["test_data"].data, np.arange(20, 40))


def test_get_deadtime_ratios():
    """Tests get_deadtime_correction_factors function."""
    # Simulate a test sectored rates dataset.
    epoch = 10
    sectored_rates_ds = xr.Dataset(
        {
            "fifo_valid_events": (["epoch"], np.random.randint(100, 200, epoch)),
            "event_active_time": (["epoch"], np.random.uniform(0, 10, epoch)),
            "start_pos": (["epoch"], np.random.randint(0, 5, epoch)),
            "start_rf": (["epoch"], np.random.randint(0, 5, epoch)),
            "start_lf": (["epoch"], np.random.randint(0, 5, epoch)),
            "coin_tn": (["epoch"], np.random.randint(0, 5, epoch)),
            "coin_bn": (["epoch"], np.random.randint(0, 5, epoch)),
            "stop_tn": (["epoch"], np.random.randint(0, 5, epoch)),
            "stop_bn": (["epoch"], np.random.randint(0, 5, epoch)),
        }
    )
    deadtime_correction_factors = get_deadtime_ratios(sectored_rates_ds)
    assert deadtime_correction_factors.shape == (sectored_rates_ds.sizes["epoch"],)
    assert np.all(deadtime_correction_factors >= 0)


def test_get_deadtime_interpolator(random_spin_data):
    """Tests get_deadtime_correction_factors function."""

    sector_rate_seconds = 20 * 60  # 20 minutes in seconds
    num_sectors = 3  # Number of sectors per pointing
    num_spins = sector_rate_seconds * num_sectors / 15  # 15 seconds per spin
    num_deadtimes = int(
        num_spins * 15
    )  # 15 sectors per spin. One deadtime ratio per sector

    deadtime_ratios = xr.DataArray(
        np.random.uniform(0.1, 1.0, num_deadtimes), dims=["epoch"]
    )
    sectored_rates_ds = xr.Dataset({"epoch": ("epoch", np.ones_like(deadtime_ratios))})
    with mock.patch(
        "imap_processing.ultra.l1c.ultra_l1c_pset_bins.get_deadtime_ratios",
        return_value=deadtime_ratios,
    ):
        deadtime_ratios = get_deadtime_ratios_by_spin_phase(sectored_rates_ds)
    np.testing.assert_array_equal(deadtime_ratios.shape, (15000))

    with mock.patch(
        "imap_processing.ultra.l1c.ultra_l1c_pset_bins.get_deadtime_ratios",
        return_value=deadtime_ratios * np.nan,
    ):
        # Assert value error is raised for NaN values
        with pytest.raises(
            ValueError,
            match="Dead time ratios contain NaN values, cannot create interpolator.",
        ):
            get_deadtime_ratios_by_spin_phase(sectored_rates_ds)


@pytest.mark.external_kernel
def test_apply_deadtime_correction(imap_ena_sim_metakernel, ancillary_files):
    """Tests apply_deadtime_correction function."""
    nside = 8
    pix = hp.nside2npix(nside)
    steps = 500  # Reduced for testing
    mock_theta = np.random.uniform(-60, 60, (pix, steps))
    mock_phi = np.random.uniform(-60, 60, (pix, steps))
    spin_phase_steps = np.zeros((pix, steps)).astype(bool)  # Spin phase steps 1-15000,
    # Simulate first 100 pixels are in the FOR for all spin phases
    inside_inds = 100
    spin_phase_steps[:inside_inds, :] = True
    deadtime_ratios = np.ones(steps)
    exposure_pointing = pd.Series(np.ones(pix))

    pixels_below_threshold = calculate_pixels_within_scattering_threshold(
        spin_phase_steps, mock_theta, mock_phi, ancillary_files, 45
    )

    exposure_pointing_adjusted = apply_deadtime_correction(
        exposure_pointing, deadtime_ratios, pixels_below_threshold
    )
    # The adjusted exposure should now be a function of pixels and energy (24)
    np.testing.assert_array_equal(exposure_pointing_adjusted.shape, (24, pix))
    # Check that the pixels inside the FOR have adjusted exposure > 1.0
    # Subset the energy dimension to check values in the last energy bin. These
    # Should have pixels that are below the FWHM scattering threshold and therefore,
    # have the exposure adjusted.
    last_energy_bin_vals = np.where(build_energy_bins()[2] >= 30)[0]
    assert np.all(exposure_pointing_adjusted[last_energy_bin_vals, :inside_inds] > 1.0)
    # Assert that pixels outside the FOR remain at 1.0
    assert np.all(exposure_pointing_adjusted[:, inside_inds:] == 1.0)


@pytest.mark.external_test_data
def test_get_spacecraft_exposure_times(
    deadtime_datasets, random_spin_data, imap_ena_sim_metakernel, ancillary_files
):
    """Test get_spacecraft_exposure_times function."""
    constant_exposure = (
        TEST_PATH / "imap_ultra_l1c-90sensor-dps-exposure_20250101_v000.csv"
    )
    steps = 500  # reduced for testing
    rates = deadtime_datasets["rates"]
    params = deadtime_datasets["params"]
    shape = 786
    df_exposure = pd.read_csv(constant_exposure)[:shape]  # Subset for testing

    pix = len(df_exposure)
    mock_theta = np.random.uniform(-60, 60, (pix, steps))
    mock_phi = np.random.uniform(-60, 60, (pix, steps))
    spin_phase_steps = np.random.randint(0, 2, (pix, steps)).astype(
        bool
    )  # Spin phase steps, random 0 or 1

    pixels_below_threshold = calculate_pixels_within_scattering_threshold(
        spin_phase_steps, mock_theta, mock_phi, ancillary_files, 45
    )
    exposure_pointing, deadtimes = get_spacecraft_exposure_times(
        df_exposure, rates, params, pixels_below_threshold
    )
    np.testing.assert_array_equal(exposure_pointing.shape, (24, shape))
    np.testing.assert_array_equal(deadtimes.shape, (15000,))


@pytest.mark.external_kernel
def test_get_helio_exposure_time_and_sensitivity(imap_ena_sim_metakernel):
    """Tests get_helio_exposure_times function."""

    start_time = 829485054.185627
    end_time = 829567884.185627

    mid_time = np.average([start_time, end_time])

    _, energy_midpoints, _ = build_energy_bins()
    nside = 128
    npix = hp.nside2npix(nside)
    shape = (len(energy_midpoints), npix)
    exposure = np.ones(shape)
    eff = np.ones(shape)
    gf = np.ones(shape)
    mock_ra = np.random.uniform(-80, 80, (npix))
    mock_dec = np.random.uniform(-80, 80, (npix))

    helio_exposure, helio_eff, helio_gf = get_helio_adjusted_data(
        mid_time, exposure, gf, eff, mock_ra, mock_dec
    )

    for helio_array, array in zip(
        [helio_exposure, helio_eff, helio_gf], [exposure, eff, gf], strict=False
    ):
        total_input = np.sum(array)
        total_output = np.sum(total_input)
        assert np.allclose(total_input, total_output, atol=1e-6)
        assert helio_array.shape == shape


def test_get_spacecraft_background_rates(
    rates_l1_test_path, use_fake_spin_data_for_time, ancillary_files
):
    "Tests calculate_background_rates function."
    # Simulate a spin table from MET = 0 to MET = 141 * 15 seconds
    use_fake_spin_data_for_time(start_met=0, end_met=141 * 15)
    df = pd.read_csv(rates_l1_test_path)

    rates = {
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
    energy_bin_edges, _, _ = build_energy_bins()
    cullingmask_spin_number = np.array([130, 131])

    background_rates = get_spacecraft_background_rates(
        rates, "ultra45", ancillary_files, energy_bin_edges, cullingmask_spin_number
    )

    assert background_rates.shape == (len(energy_bin_edges), hp.nside2npix(128))
    assert np.allclose(background_rates[0, :], np.full((196608,), 6.37052558e-11))


def test_rate_uncertainty():
    """Tests spacecraft_count_rate_uncertainty function."""

    hist = np.array(
        [[0.0, 1.0, 4.0], [9.0, 16.0, 25.0], [36.0, 49.0, 64.0], [0.0, 100.0, 121.0]]
    )

    exposure = np.ones_like(hist)
    uncertainty = get_spacecraft_count_rate_uncertainty(hist, exposure)
    expected = np.sqrt(hist)

    np.testing.assert_allclose(uncertainty, expected, atol=1e-6)
