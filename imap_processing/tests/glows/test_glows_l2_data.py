import numpy as np
import pytest
import xarray as xr

from imap_processing.glows.l1b.glows_l1b_data import PipelineSettings
from imap_processing.glows.l2.glows_l2_data import DailyLightcurve, HistogramL2


@pytest.fixture
def pipeline_settings():
    """PipelineSettings with flags matching the default pipeline settings JSON.

    active_bad_time_flags has 17 entries (is_night and
    is_spin_period_difference_beyond_threshold are False, all others True).
    active_bad_angle_flags has 4 entries (all True).
    """
    active_bad_time_flags = [
        True,  # is_pps_missing
        True,  # is_time_status_missing
        True,  # is_phase_missing
        True,  # is_spin_period_missing
        True,  # is_overexposed
        True,  # is_direct_event_non_monotonic
        False,  # is_night
        True,  # is_hv_test_in_progress
        True,  # is_test_pulse_in_progress
        True,  # is_memory_error_detected
        True,  # is_generated_on_ground
        True,  # is_beyond_daily_statistical_error
        True,  # is_temperature_std_dev_beyond_threshold
        True,  # is_hv_voltage_std_dev_beyond_threshold
        True,  # is_spin_period_std_dev_beyond_threshold
        True,  # is_pulse_length_std_dev_beyond_threshold
        False,  # is_spin_period_difference_beyond_threshold
    ]
    active_bad_angle_flags = [
        True,  # is_close_to_uv_source
        True,  # is_inside_excluded_region
        True,  # is_excluded_by_instr_team
        True,  # is_suspected_transient
    ]
    pipeline_dataset = xr.Dataset(
        {
            "active_bad_time_flags": xr.DataArray(active_bad_time_flags),
            "active_bad_angle_flags": xr.DataArray(active_bad_angle_flags),
        }
    )
    return PipelineSettings(pipeline_dataset)


@pytest.fixture
def l1b_dataset():
    """Minimal L1B dataset for testing DailyLightcurve.

    Two timestamps, four bins.
    Bin 3 is masked (-1) at timestamp 0.
    """
    n_epochs, n_bins = 2, 4
    epoch = xr.DataArray(np.arange(n_epochs), dims=["epoch"])
    bins = xr.DataArray(np.arange(n_bins), dims=["bins"])

    histogram = np.array([[10, 20, 30, -1], [10, 20, 30, 40]], dtype=float)
    spin_angle = np.tile(np.linspace(0, 270, n_bins), (n_epochs, 1))

    ds = xr.Dataset(
        {
            "histogram": (["epoch", "bins"], histogram),
            "spin_period_average": (["epoch"], [15.0, 15.0]),
            "number_of_spins_per_block": (["epoch"], [5, 5]),
            "imap_spin_angle_bin_cntr": (["epoch", "bins"], spin_angle),
        },
        coords={"epoch": epoch, "bins": bins},
    )
    return ds


def test_photon_flux(l1b_dataset):
    """Flux = sum(histograms) / sum(exposure_times) per bin (Eq. 50)."""
    lc = DailyLightcurve(l1b_dataset)

    # l1b_exposure_time_per_bin = spin_period_average *
    # number_of_spins_per_block / number_of_bins_per_histogram
    exposure_per = 15.0 * 5 / 4
    expected_raw = np.array([20, 40, 60, 40])
    # Exposure accumulates uniformly per good-time file regardless of per-bin masking
    expected_exposure = np.array(
        [2 * exposure_per, 2 * exposure_per, 2 * exposure_per, 2 * exposure_per]
    )
    expected_flux = expected_raw / expected_exposure

    assert np.allclose(lc.raw_histograms, expected_raw)
    assert np.allclose(lc.exposure_times, expected_exposure)
    assert np.allclose(lc.photon_flux, expected_flux)


def test_flux_uncertainty(l1b_dataset):
    """Uncertainty = sqrt(sum_hist) / exposure per bin (Eq. 54)."""
    lc = DailyLightcurve(l1b_dataset)

    expected_uncertainty = np.sqrt(lc.raw_histograms) / lc.exposure_times
    assert np.allclose(lc.flux_uncertainties, expected_uncertainty)


def test_zero_exposure_bins():
    """Bins with all-masked histograms get zero flux and uncertainty.

    Exposure time still accumulates uniformly from each good-time file even
    when all histogram values are masked (-1). Flux and uncertainty are zero
    because the raw histogram sums are zero.
    """
    n_epochs, n_bins = 2, 3
    histogram = np.full((n_epochs, n_bins), -1, dtype=float)
    spin_angle = np.tile(np.linspace(0, 240, n_bins), (n_epochs, 1))

    ds = xr.Dataset(
        {
            "histogram": (["epoch", "bins"], histogram),
            "spin_period_average": (["epoch"], [15.0, 15.0]),
            "number_of_spins_per_block": (["epoch"], [5, 5]),
            "imap_spin_angle_bin_cntr": (["epoch", "bins"], spin_angle),
        },
        coords={"epoch": xr.DataArray(np.arange(n_epochs), dims=["epoch"])},
    )
    lc = DailyLightcurve(ds)

    expected_exposure = 2 * 15.0 * 5 / 3
    assert np.all(lc.photon_flux == 0)
    assert np.all(lc.flux_uncertainties == 0)
    assert np.allclose(lc.exposure_times, expected_exposure)


def test_number_of_bins(l1b_dataset):
    lc = DailyLightcurve(l1b_dataset)
    assert lc.number_of_bins == 4
    assert len(lc.spin_angle) == 4
    assert len(lc.photon_flux) == 4
    assert len(lc.flux_uncertainties) == 4
    assert len(lc.exposure_times) == 4


def test_filter_good_times():
    """Epochs where any active flag is 0 are excluded; inactive flags are ignored."""
    active_flags = np.ones((17,))
    active_flags[16] = 0  # flag 16 is inactive
    test_flags = np.ones((4, 17))
    test_flags[1, 0] = 0  # epoch 1 fails active flag 0 -> bad time
    test_flags[3, 16] = 0  # epoch 3 fails inactive flag 16 -> still good time
    flags = xr.DataArray(test_flags, dims=["epoch", "flags"])

    good_times = HistogramL2.return_good_times(flags, active_flags)
    expected_good_times = [0, 2, 3]

    assert np.array_equal(good_times, expected_good_times)
