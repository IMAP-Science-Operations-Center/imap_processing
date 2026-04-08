from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing.glows.l1b.glows_l1b_data import PipelineSettings
from imap_processing.glows.l2.glows_l2_data import DailyLightcurve, HistogramL2
from imap_processing.glows.utils.constants import GlowsConstants
from imap_processing.spice.time import met_to_sclkticks, sct_to_et


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
            "active_bad_time_flags": xr.DataArray(
                active_bad_time_flags, dims=["time_flag_index"]
            ),
            "active_bad_angle_flags": xr.DataArray(
                active_bad_angle_flags, dims=["angle_flag_index"]
            ),
        }
    )
    return PipelineSettings(pipeline_dataset)


@pytest.fixture
def l1b_dataset():
    """Minimal L1B dataset for testing DailyLightcurve.

    Two timestamps, four bins.
    Bin 3 is masked (HISTOGRAM_FILLVAL) at timestamp 0.
    histogram_flag_array has shape (epoch, bad_angle_flags, bins) with all zeros.
    """
    n_epochs, n_bins, n_flags = 2, 4, 4
    fillval = GlowsConstants.HISTOGRAM_FILLVAL
    epoch = xr.DataArray(np.arange(n_epochs), dims=["epoch"])
    bins = xr.DataArray(np.arange(n_bins), dims=["bins"])

    histogram = np.array([[10, 20, 30, fillval], [10, 20, 30, 40]], dtype=float)
    spin_angle = np.tile(np.linspace(0, 270, n_bins), (n_epochs, 1))
    histogram_flag_array = np.zeros((n_epochs, n_flags, n_bins), dtype=np.uint8)

    ds = xr.Dataset(
        {
            "histogram": (["epoch", "bins"], histogram),
            "spin_period_average": (["epoch"], [15.0, 15.0]),
            "number_of_spins_per_block": (["epoch"], [5, 5]),
            "imap_spin_angle_bin_cntr": (["epoch", "bins"], spin_angle),
            "imap_start_time": (["epoch"], [0.0, 1.0]),
            "histogram_flag_array": (
                ["epoch", "bad_angle_flags", "bins"],
                histogram_flag_array,
            ),
            "number_of_bins_per_histogram": (["epoch"], [n_bins, n_bins]),
        },
        coords={"epoch": epoch, "bins": bins},
    )
    return ds


def test_get_calibration_factor(mock_calibration_dataset):
    """Test selecting correct calibration factor.

    Mock calibration data:
      start_time_utc (dims epoch × start_time_utc_dim_0, same per epoch):
          ["2011-09-19T09:58:04", "2011-09-20T18:12:48", "2011-09-21T18:15:50"]
      cps_per_r (dims epoch × cps_per_r_dim_0, same per epoch):
          index 0 → 0.849,  index 1 → 1.020,  index 2 → 1.500
    """
    # Case 1: The mid-epoch ('2011-09-22T10:30:55.015') falls after the
    # start_time_utc entries, so the last entry (index 2) is selected → 1.500.

    # ["2011-09-22T07:45:55.015", "2011-09-22T10:30:55.015", "2011-09-22T13:15:55.015"]
    later_epoch = np.array([369949621199000000, 369959521199000000, 369969421199000000])
    assert HistogramL2.get_calibration_factor(
        later_epoch, mock_calibration_dataset
    ) == pytest.approx(1.500)

    # Case 2: The mid-epoch ('2011-09-21T00:52:15.000') falls between the 2nd and
    # 3rd start_time_utc entries, so the 2nd entry (index 1) is selected → 1.020.

    # ['2011-09-21T00:50:15.000', '2011-09-21T00:52:15.000', '2011-09-21T00:54:15.000']
    between_epoch = np.array(
        [369838281184000000, 369838401184000000, 369838521184000000]
    )
    assert HistogramL2.get_calibration_factor(
        between_epoch, mock_calibration_dataset
    ) == pytest.approx(1.020)

    # Case 3: The mid-epoch is before all start_time_utc entries,
    # so a KeyError is raised by xarray's "pad" selection method.

    # ['2011-09-18T19:59:08.816', '2011-09-18T20:01:08.816', '2011-09-18T20:03:08.816']
    early_epoch = np.array([369648015000000000, 369648135000000000, 369648255000000000])
    with pytest.raises(KeyError):
        HistogramL2.get_calibration_factor(early_epoch, mock_calibration_dataset)


@pytest.mark.external_kernel
def test_ecliptic_coords_computation(furnish_kernels):
    """Test method that computes ecliptic coordinates."""

    # Use a met value within the SPICE kernel coverage (2026-01-01).
    data_start_time_et = sct_to_et(met_to_sclkticks(504975603.125))
    n_bins = 4
    spin_angle = np.linspace(0, 270, n_bins)

    kernels = [
        "naif0012.tls",
        "imap_sclk_0000.tsc",
        "imap_130.tf",
        "imap_science_120.tf",
        "sim_1yr_imap_pointing_frame.bc",
    ]

    with furnish_kernels(kernels):
        ecliptic_lon, ecliptic_lat = (
            DailyLightcurve.compute_ecliptic_coords_of_bin_centers(
                data_start_time_et, spin_angle
            )
        )

    # ecliptic_lon and ecliptic_lat must have one entry per bin
    assert len(ecliptic_lon) == n_bins
    assert len(ecliptic_lat) == n_bins

    # ecliptic longitude must be in [0, 360)
    assert np.all(ecliptic_lon >= 0.0)
    assert np.all(ecliptic_lon < 360.0)

    # ecliptic latitude must be in [-90, 90]
    assert np.all(ecliptic_lat >= -90.0)
    assert np.all(ecliptic_lat <= 90.0)

    # values must be finite (no NaN / Inf from SPICE)
    assert np.all(np.isfinite(ecliptic_lon))
    assert np.all(np.isfinite(ecliptic_lat))


def test_photon_flux(l1b_dataset, mock_ecliptic_bin_centers):
    """
    Flux = (sum(histograms) / sum(exposure_times)) /
            Rayleigh calibration factor

            per bin (Eq. 50-53)
    """
    mock_cal_factor = 2
    lc = DailyLightcurve(
        l1b_dataset, position_angle=0.0, calibration_factor=mock_cal_factor
    )

    # l1b_exposure_time_per_bin = spin_period_average *
    # number_of_spins_per_block / number_of_bins_per_histogram
    exposure_per = 15.0 * 5 / 4
    expected_raw = np.array([20, 40, 60, 40])
    # Exposure accumulates uniformly per good-time file regardless of per-bin masking
    expected_exposure = np.array(
        [2 * exposure_per, 2 * exposure_per, 2 * exposure_per, 2 * exposure_per]
    )
    expected_flux = (expected_raw / expected_exposure) / mock_cal_factor

    assert np.allclose(lc.raw_histograms, expected_raw)
    assert np.allclose(lc.exposure_times, expected_exposure)
    assert np.allclose(lc.photon_flux, expected_flux)


def test_flux_uncertainty(l1b_dataset, mock_ecliptic_bin_centers):
    """
    Uncertainty = sqrt(sum_hist) / exposure /
                Rayleigh calibration factor

                per bin (Eq. 54-55)."""
    mock_cal_factor = 2
    lc = DailyLightcurve(
        l1b_dataset, position_angle=0.0, calibration_factor=mock_cal_factor
    )

    expected_uncertainty = (
        np.sqrt(lc.raw_histograms) / lc.exposure_times
    ) / mock_cal_factor
    assert np.allclose(lc.flux_uncertainties, expected_uncertainty)


def test_zero_exposure_bins(l1b_dataset, mock_ecliptic_bin_centers):
    """Bins with all-masked histograms get zero flux and uncertainty.

    Exposure time still accumulates uniformly from each good-time file even
    when all histogram values are masked (HISTOGRAM_FILLVAL). Flux and
    uncertainty are zero because the raw histogram sums are zero.
    """
    mock_cal_factor = 1
    l1b_dataset["histogram"].values[:] = GlowsConstants.HISTOGRAM_FILLVAL
    lc = DailyLightcurve(
        l1b_dataset, position_angle=0.0, calibration_factor=mock_cal_factor
    )

    expected_exposure = 2 * 15.0 * 5 / 4
    assert np.all(lc.photon_flux == 0)
    assert np.all(lc.flux_uncertainties == 0)
    assert np.allclose(lc.exposure_times, expected_exposure)


def test_zero_exposure_values(l1b_dataset, mock_ecliptic_bin_centers):
    """Zero exposure yields zero flux and zero uncertainty per bin."""

    # Note: all bins have the same exposure time, so if one is zero all are zero.

    # Update values used to calculate exposure times to
    # ensure a zero exposure result.
    l1b_dataset["spin_period_average"].data[:] = 0
    l1b_dataset["number_of_spins_per_block"].data[:] = 0

    mock_cal_factor = 1

    with np.errstate(divide="raise", invalid="raise"):
        lc = DailyLightcurve(
            l1b_dataset, position_angle=0.0, calibration_factor=mock_cal_factor
        )

    expected = np.zeros(l1b_dataset.sizes["bins"], dtype=float)
    assert lc.exposure_times.shape == expected.shape
    assert len(np.unique(lc.exposure_times)) == 1
    assert np.array_equal(lc.exposure_times, expected)
    assert np.array_equal(lc.photon_flux, expected)
    assert np.array_equal(lc.flux_uncertainties, expected)
    assert np.all(np.isfinite(lc.photon_flux))
    assert np.all(np.isfinite(lc.flux_uncertainties))


def test_number_of_bins(l1b_dataset, mock_ecliptic_bin_centers):
    mock_cal_factor = 1
    lc = DailyLightcurve(
        l1b_dataset, position_angle=0.0, calibration_factor=mock_cal_factor
    )
    assert lc.number_of_bins == 4
    assert len(lc.spin_angle) == 4
    assert len(lc.photon_flux) == 4
    assert len(lc.flux_uncertainties) == 4
    assert len(lc.exposure_times) == 4
    assert len(lc.ecliptic_lon) == 4
    assert len(lc.ecliptic_lat) == 4


def test_histogram_flag_array_or_propagation(l1b_dataset, mock_ecliptic_bin_centers):
    """histogram_flag_array is OR'd across all L1B epochs and flag rows per bin.

    Per Section 12.3.4: a flag is True in L2 if it is True in any L1B block.
    """
    # epoch 0, flag row 0 (IS_CLOSE_TO_UV_SOURCE=1): bin 0 flagged
    # epoch 1, flag row 1 (IS_INSIDE_EXCLUDED_REGION=2): bin 2 flagged
    # epoch 1, flag row 0 (IS_CLOSE_TO_UV_SOURCE=2): bin 0 flagged
    l1b_dataset["histogram_flag_array"].values[0, 0, 0] = 1
    l1b_dataset["histogram_flag_array"].values[1, 1, 2] = 2
    l1b_dataset["histogram_flag_array"].values[1, 0, 0] = 2

    mock_cal_factor = 1
    lc = DailyLightcurve(
        l1b_dataset, position_angle=0.0, calibration_factor=mock_cal_factor
    )

    assert (
        lc.histogram_flag_array[0] == 3
    )  # IS_CLOSE_TO_UV_SOURCE and IS_INSIDE_EXCLUDED_REGION
    assert lc.histogram_flag_array[1] == 0  # no flags on bin 1
    assert lc.histogram_flag_array[2] == 2  # IS_INSIDE_EXCLUDED_REGION
    assert lc.histogram_flag_array[3] == 0  # no flags on bin 3


def test_histogram_flag_array_zero_epochs(mock_ecliptic_bin_centers):
    """histogram_flag_array is all zeros when the input dataset is empty.

    Note: this is NEVER expected to happen in production
    """
    n_bins, n_flags = 4, 4
    histogram = np.empty((0, n_bins), dtype=float)
    spin_angle = np.empty((0, n_bins), dtype=float)
    histogram_flag_array = np.empty((0, n_flags, n_bins), dtype=np.uint8)

    ds = xr.Dataset(
        {
            "histogram": (["epoch", "bins"], histogram),
            "spin_period_average": (["epoch"], []),
            "number_of_spins_per_block": (["epoch"], []),
            "imap_spin_angle_bin_cntr": (["epoch", "bins"], spin_angle),
            "histogram_flag_array": (
                ["epoch", "bad_angle_flags", "bins"],
                histogram_flag_array,
            ),
            "number_of_bins_per_histogram": (["epoch"], []),
        },
        coords={"epoch": xr.DataArray(np.arange(0), dims=["epoch"])},
    )
    mock_cal_factor = 1
    lc = DailyLightcurve(ds, position_angle=0.0, calibration_factor=mock_cal_factor)

    # if the dataset is empty, there is no way to infer the number_of_bins
    assert len(lc.histogram_flag_array) == 0
    assert np.all(lc.histogram_flag_array == 0)


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


# ── spin_angle tests ──────────────────────────────────────────────────────────


def test_spin_angle_offset_formula(l1b_dataset, mock_ecliptic_bin_centers):
    """spin_angle = (imap_spin_angle_bin_cntr - position_angle + 360) % 360.

    Fixture spin_angle_bin_cntr = [0, 90, 180, 270], position_angle = 90.
    Expected before roll: [270, 0, 90, 180].
    Minimum is at index 1, so roll = -1 -> [0, 90, 180, 270].
    """
    mock_cal_factor = 1
    lc = DailyLightcurve(
        l1b_dataset, position_angle=90.0, calibration_factor=mock_cal_factor
    )
    expected = np.array([0.0, 90.0, 180.0, 270.0])
    assert np.allclose(lc.spin_angle, expected)


def test_spin_angle_starts_at_minimum(l1b_dataset, mock_ecliptic_bin_centers):
    """After rolling, lc.spin_angle[0] is the minimum value.

    Fixture spin_angle_bin_cntr = [0, 90, 180, 270], position_angle = 45.
    Before roll: [315, 45, 135, 225]; minimum 45 is at index 1 -> roll = -1
    -> [45, 135, 225, 315].
    """
    mock_cal_factor = 1
    lc = DailyLightcurve(
        l1b_dataset, position_angle=45.0, calibration_factor=mock_cal_factor
    )
    assert lc.spin_angle[0] == np.min(lc.spin_angle)
    assert np.allclose(lc.spin_angle, np.array([45.0, 135.0, 225.0, 315.0]))


# ── position_angle_offset_average tests ──────────────────────────────────────


def test_compute_position_angle():
    """compute_position_angle returns (360 - azimuth) % 360 (Eq. 30)."""
    target_module = (
        "imap_processing.glows.l2.glows_l2_data.get_instrument_mounting_az_el"
    )
    with patch(target_module, return_value=(270.0, 0.0)):
        result = HistogramL2.compute_position_angle(None)
    assert result == pytest.approx(90.0)


@pytest.fixture
def l1b_dataset_full():
    """Minimal L1B dataset with all variables required by HistogramL2.

    Two epochs, four bins, 17 flags (all good).
    """
    n_epochs, n_bins, n_angle_flags, n_time_flags = 2, 4, 4, 17
    fillval = GlowsConstants.HISTOGRAM_FILLVAL
    epoch = np.arange(n_epochs, dtype=np.float64)

    histogram = np.array([[10, 20, 30, fillval], [10, 20, 30, 40]], dtype=float)
    spin_angle = np.tile(np.linspace(0, 270, n_bins), (n_epochs, 1))
    histogram_flag_array = np.zeros((n_epochs, n_angle_flags, n_bins), dtype=np.uint8)

    return xr.Dataset(
        {
            "histogram": (["epoch", "bins"], histogram),
            "spin_period_average": (["epoch"], [15.0, 15.0]),
            "number_of_spins_per_block": (["epoch"], [5, 5]),
            "imap_spin_angle_bin_cntr": (["epoch", "bins"], spin_angle),
            "histogram_flag_array": (
                ["epoch", "bad_angle_flags", "bins"],
                histogram_flag_array,
            ),
            "number_of_bins_per_histogram": (["epoch"], [n_bins, n_bins]),
            "flags": (
                ["epoch", "flag_index"],
                np.ones((n_epochs, n_time_flags)),
            ),
            "filter_temperature_average": (["epoch"], [20.0, 21.0]),
            "hv_voltage_average": (["epoch"], [1000.0, 1000.0]),
            "pulse_length_average": (["epoch"], [5.0, 5.0]),
            "spin_period_ground_average": (["epoch"], [15.0, 15.0]),
            "spacecraft_location_average": (
                ["epoch", "xyz"],
                np.ones((n_epochs, 3)),
            ),
            "spacecraft_velocity_average": (
                ["epoch", "xyz"],
                np.ones((n_epochs, 3)),
            ),
            "spin_axis_orientation_average": (
                ["epoch", "lonlat"],
                np.ones((n_epochs, 2)),
            ),
            "imap_start_time": (["epoch"], [0.0, 1.0]),
            "imap_time_offset": (["epoch"], [60.0, 60.0]),
        },
        coords={"epoch": xr.DataArray(epoch, dims=["epoch"])},
    )


def test_position_angle_offset_average(
    l1b_dataset_full,
    pipeline_settings,
    mock_ecliptic_bin_centers,
    mock_calibration_dataset,
):
    """position_angle_offset_average is a scalar equal to the result of
    compute_position_angle (Eq. 30, Section 10.6). It is constant across the
    observational day since it depends only on instrument mounting geometry.
    """
    mock_pa = 42.5
    mock_cal_factor = 1

    with (
        patch.object(HistogramL2, "compute_position_angle", return_value=mock_pa),
        patch.object(
            HistogramL2, "get_calibration_factor", return_value=mock_cal_factor
        ),
    ):
        l2 = HistogramL2(l1b_dataset_full, pipeline_settings, mock_calibration_dataset)

        assert l2.position_angle_offset_average == pytest.approx(mock_pa)
