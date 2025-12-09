"Tests pointing sets"

from unittest import mock

import astropy_healpix.healpy as hp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import interpolate

from imap_processing import imap_module_directory
from imap_processing.ultra.l1c import ultra_l1c_pset_bins
from imap_processing.ultra.l1c.spacecraft_pset import (
    calculate_fwhm_spun_scattering,
)
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    calculate_exposure_time,
    get_deadtime_ratios,
    get_deadtime_ratios_by_spin_phase,
    get_energy_delta_minus_plus,
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

    assert energy_bin_start[0] == 3
    assert np.allclose(energy_bin_start[1], 3.4, atol=1e-3)
    assert len(intervals) == 46
    assert energy_midpoints[0] == (energy_bin_start[0] + energy_bin_end[0]) / 2

    # Comparison to expected values.
    np.testing.assert_allclose(energy_bin_end[1], 3.8)
    np.testing.assert_allclose(energy_bin_start[-1], 286.208, atol=1e-3)
    np.testing.assert_allclose(energy_bin_end[-1], 316.334, atol=1e-3)

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

    def mock_build_energy_bins(energy_bins=None):
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

    # Spot check that 2 counts are in the second energy bin
    assert np.sum(hist[2, :]) == 2

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
        np.arange(
            10, 20
        ),  # Make sure duplicate epochs with the same mode are filtered out
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
        deadtime_ratios = get_deadtime_ratios_by_spin_phase(
            sectored_rates_ds, spin_steps=num_deadtimes
        )
    np.testing.assert_array_equal(deadtime_ratios.shape, (num_deadtimes))

    with mock.patch(
        "imap_processing.ultra.l1c.ultra_l1c_pset_bins.get_deadtime_ratios",
        return_value=deadtime_ratios * np.nan,
    ):
        # Assert value error is raised for NaN values
        with pytest.raises(
            ValueError,
            match="All dead time ratios are NaN, cannot interpolate",
        ):
            get_deadtime_ratios_by_spin_phase(
                sectored_rates_ds, spin_steps=num_deadtimes
            )


@pytest.mark.external_test_data
def test_get_deadtime_interpolator_no_sectored_rates(ancillary_files):
    """Tests get_deadtime_correction_factors function."""

    num_deadtimes = 15000  # Standard number of spin phases
    sensor = 45
    # If the sectored rates dataset is None, the function should use the
    # static deadtime ratios lookup.
    dt_ratios = get_deadtime_ratios_by_spin_phase(
        sectored_rates=None,
        spin_steps=num_deadtimes,
        sensor_id=sensor,
        ancillary_files=ancillary_files,
    )
    spin_phase, dts = ultra_l1c_pset_bins.get_static_deadtime_ratios(
        sensor, ancillary_files
    )
    # Calculate the nominal spin phases at the supplied resolution and query the pchip
    # interpolator to get the deadtime ratios.
    nominal_spin_phases = np.arange(0, 360, 360 / num_deadtimes)
    expected_dt_ratios = interpolate.PchipInterpolator(spin_phase, dts)(
        nominal_spin_phases
    )
    np.testing.assert_array_equal(dt_ratios, expected_dt_ratios)


@pytest.mark.external_kernel
def test_apply_deadtime_correction(imap_ena_sim_metakernel, ancillary_files):
    """Tests apply_deadtime_correction function."""
    nside = 8
    pix = hp.nside2npix(nside)
    steps = 500  # Reduced for testing
    np.random.seed(42)
    mock_theta = np.random.uniform(-60, 60, (steps, pix))
    mock_phi = np.random.uniform(-60, 60, (steps, pix))
    spin_phase_steps = xr.DataArray(
        np.zeros((steps, pix)).astype(bool), dims=("spin_phase_step", "pixel")
    )
    # Simulate first 100 pixels are in the FOR for all spin phases
    inside_inds = 100
    spin_phase_steps[:, :inside_inds] = True
    deadtime_ratios = xr.DataArray(np.ones(steps), dims="spin_phase_step")

    valid_spun_pixels, fwhm_theta, fwhm_phi, thresholds = (
        calculate_fwhm_spun_scattering(
            spin_phase_steps,
            mock_theta,
            mock_phi,
            ancillary_files,
            45,
            reject_scattering=False,
        )
    )
    boundary_sf = xr.DataArray(np.ones((pix, steps)), dims=("pixel", "spin_phase_step"))
    exposure_pointing_adjusted = calculate_exposure_time(
        deadtime_ratios,
        valid_spun_pixels,
        boundary_sf,
        apply_bsf=True,
    )
    # The adjusted exposure should be of shape (46,npix)
    np.testing.assert_array_equal(exposure_pointing_adjusted.shape, (46, pix))
    # Check that the pixels inside the FOR have adjusted exposure > 0.
    assert np.all(exposure_pointing_adjusted[:, :inside_inds] > 0)
    # Assert that pixels outside the FOR remain at 0.
    assert np.all(exposure_pointing_adjusted[:, inside_inds:] == 0)


@pytest.mark.external_kernel
def test_apply_deadtime_correction_energy_dep(imap_ena_sim_metakernel, ancillary_files):
    """Tests apply_deadtime_correction function when scattering rejection is on."""
    nside = 8
    pix = hp.nside2npix(nside)
    steps = 500  # Reduced for testing
    np.random.seed(42)
    mock_theta = np.random.uniform(-60, 60, (steps, pix))
    mock_phi = np.random.uniform(-60, 60, (steps, pix))
    spin_phase_steps = xr.DataArray(
        np.zeros((steps, pix)).astype(bool), dims=("spin_phase_step", "pixel")
    )
    # Simulate first 100 pixels are in the FOR for all spin phases
    inside_inds = 100
    spin_phase_steps[:, :inside_inds] = True
    deadtime_ratios = xr.DataArray(np.ones(steps), dims="spin_phase_step")

    valid_spun_pixels, fwhm_theta, fwhm_phi, thresholds = (
        calculate_fwhm_spun_scattering(
            spin_phase_steps,
            mock_theta,
            mock_phi,
            ancillary_files,
            45,
            reject_scattering=True,
        )
    )
    boundary_sf = xr.DataArray(np.ones((steps, pix)), dims=("spin_phase_step", "pixel"))
    exposure_pointing_adjusted = calculate_exposure_time(
        deadtime_ratios,
        valid_spun_pixels,
        boundary_sf,
        apply_bsf=True,
    )
    # The adjusted exposure should be of shape (46,npix)
    np.testing.assert_array_equal(exposure_pointing_adjusted.shape, (46, pix))
    # Check that the pixels inside the FOR have adjusted exposure > 0.
    # Subset the energy dimension to check values in the last energy bin. These
    # Should have pixels that are below the FWHM scattering threshold and therefore,
    # have the exposure adjusted.
    last_energy_bin_vals = np.where(build_energy_bins()[2] >= 40)[0]
    assert np.all(exposure_pointing_adjusted[last_energy_bin_vals, :inside_inds] > 0)
    # Assert that pixels outside the FOR remain at 0.
    assert np.all(exposure_pointing_adjusted[:, inside_inds:] == 0)


@pytest.mark.external_test_data
def test_get_spacecraft_exposure_times(
    deadtime_datasets,
    random_spin_data,
    imap_ena_sim_metakernel,
    ancillary_files,
    use_fake_spin_data_for_time,
):
    """Test get_spacecraft_exposure_times function."""
    data_start_time = 453051293.0
    data_end_time = 453070000.0
    use_fake_spin_data_for_time(data_start_time, data_end_time)
    steps = 500  # reduced for testing
    rates = deadtime_datasets["rates"]
    params = deadtime_datasets["params"]

    pix = 786
    mock_theta = np.random.uniform(-60, 60, (steps, pix))
    mock_phi = np.random.uniform(-60, 60, (steps, pix))
    np.random.seed(42)
    spin_phase_steps = xr.DataArray(
        np.random.randint(0, 2, (steps, pix)).astype(bool),
        dims=("spin_phase_step", "pixel"),
    )  # Spin phase steps, random 0 or 1

    pixels_below_threshold, fwhm_theta, fwhm_phi, thresholds = (
        calculate_fwhm_spun_scattering(
            spin_phase_steps, mock_theta, mock_phi, ancillary_files, 45
        )
    )
    boundary_sf = xr.DataArray(np.ones((steps, pix)), dims=("spin_phase_step", "pixel"))
    exposure_pointing, deadtimes = get_spacecraft_exposure_times(
        rates,
        params,
        pixels_below_threshold,
        boundary_sf,
        (
            data_start_time,
            data_start_time,
        ),
        46,  # number of energy bins
        pix,
    )
    np.testing.assert_array_equal(exposure_pointing.shape, (46, pix))
    np.testing.assert_array_equal(deadtimes.shape, (steps,))


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
    goodtimes_spin_number = np.array([130, 131])

    background_rates = get_spacecraft_background_rates(
        rates, "ultra45", ancillary_files, energy_bin_edges, goodtimes_spin_number
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
