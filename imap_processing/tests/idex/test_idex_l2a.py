"""Tests the L2a processing for IDEX data"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import exponnorm

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.idex import idex_constants
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.idex.idex_l2a import (
    BaselineNoiseTime,
    analyze_peaks,
    butter_lowpass_filter,
    calculate_kappa,
    calculate_snr,
    calculate_velocity_and_mass,
    chi_square,
    estimate_dust_mass,
    fit_impact,
    idex_l2a,
    invert_rise_time_to_velocity,
    load_calibration_files,
    log_smooth_powerlaw,
    remove_signal_noise,
    sine_fit,
    time_to_mass,
)
from imap_processing.idex.idex_utils import get_idex_attrs


@pytest.fixture
def l2a_dataset(
    l1b_dataset: xr.Dataset, decom_test_data_sci, ancillary_files, _download_test_data
) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.
    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    idex_attrs = get_idex_attrs("l1b")
    spin_phase_angles = xr.DataArray(
        np.random.uniform(0.0, 360.0, len(l1b_dataset.epoch)),
        dims="epoch",
        attrs=idex_attrs.get_variable_attributes("spin_phase"),
    )
    with mock.patch(
        "imap_processing.idex.idex_l1b.get_spice_data",
        return_value={"spin_phase": spin_phase_angles},
    ):
        dataset = idex_l2a(idex_l1b(decom_test_data_sci, "sci-10days"), ancillary_files)
    return dataset


def mock_microphonics_noise(time: np.ndarray) -> np.ndarray:
    """Function to mock signal noise (linear and sine wave) due to microphonics."""
    noise_frequency = idex_constants.TARGET_NOISE_FREQUENCY
    phase_shift = 45
    amp = 10
    # Create a sine wave signal
    sine_signal = sine_fit(time, amp, noise_frequency, phase_shift)
    # Combine the sine wave signals with a linear signal to create noise
    combined_sig = sine_signal + (time * 5)

    return combined_sig


def _write_calibration_csv(path, values):
    """Write a one-row calibration CSV with the ancillary-file structure."""
    header = "A,a1,a2,a3,v_b,v_c,k,sigma,delta\n"
    row = ",".join(str(value) for value in values) + "\n"
    path.write_text(header + row)


@pytest.mark.external_test_data
def test_l2a_logical_source_and_cdf(l2a_dataset: xr.Dataset):
    """Tests that the ``idex_l2a`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    l2a_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_src = "imap_idex_l2a_sci-10days"
    assert l2a_dataset.attrs["Logical_source"] == expected_src
    # Verify the CDF file can be created with no errors.
    l2a_dataset.attrs["Data_version"] = "999"
    file_name = write_cdf(l2a_dataset)
    assert file_name.exists()
    assert file_name.name == "imap_idex_l2a_sci-10days_20231218_v999.cdf"
    ds = load_cdf(file_name)
    spin_phase = ds["spin_phase"].values
    spin_phase_attrs = ds["spin_phase"].attrs
    assert spin_phase.dtype == np.float64
    assert np.isclose(spin_phase_attrs["FILLVAL"], np.float64(-1.0e31))

    expected_vars = [
        "tof_snr",
        "tof_peak_kappa",
        "mass_scale",
        "target_low_fit_parameters",
        "target_low_impact_charge",
        "target_low_dust_mass_estimate",
        "target_low_velocity_estimate",
        "target_low_chi_squared",
        "target_low_reduced_chi_squared",
        "target_low_fit_results",
        "target_high_fit_parameters",
        "target_high_impact_charge",
        "target_high_dust_mass_estimate",
        "target_high_velocity_estimate",
        "target_high_chi_squared",
        "target_high_reduced_chi_squared",
        "target_high_fit_results",
        "ion_grid_fit_parameters",
        "ion_grid_impact_charge",
        "ion_grid_dust_mass_estimate",
        "ion_grid_velocity_estimate",
        "ion_grid_chi_squared",
        "ion_grid_reduced_chi_squared",
        "ion_grid_fit_results",
        "tof_peak_fit_parameters",
        "tof_peak_area_under_fit",
        "tof_peak_chi_square",
        "tof_peak_reduced_chi_square",
        "tof_peak_kappa",
        "tof_snr",
        "mass",
    ]

    cdf_vars = l2a_dataset.variables
    for var in expected_vars:
        assert var in cdf_vars
    for var in l2a_dataset.data_vars:
        assert "DICT_KEY" in l2a_dataset[var].attrs, (
            f"Variable {var} is missing the DICT_KEY attribute for SPASE metadata."
        )

    # TODO: remove this NAN block when fitting logic is applied
    expected_nan_vars = [
        "ion_grid_dust_mass_estimate",
        "ion_grid_velocity_estimate",
        "tof_peak_area_under_fit",
        "tof_peak_chi_square",
        "tof_peak_fit_parameters",
        "tof_peak_kappa",
        "tof_peak_reduced_chi_square",
        "tof_snr",
        "mass",
        "mass_scale",
    ]
    for var in expected_nan_vars:
        assert np.isnan(l2a_dataset[var].data).all(), (
            f"Variable {var} should be fully NaN for the temporary L2A patch."
        )


def test_time_to_mass_zero_lag():
    """
    Tests that the time_to_mass function correctly converts time-of-flight
    to a mass scale using known peak positions.
    """
    carbon_mass = 12
    masses = np.asarray([1, 4, 9])

    expected_lag = 10
    expected_stretch = 1500
    # Create a 2d time of flight array exactly where we would expect the peaks to be
    # Each mass should appear at time t = 1400 * sqrt(m) ns
    tof = np.zeros((15, int(np.sqrt(masses[-1]) * expected_stretch + 1 + expected_lag)))
    min_stretch = 1400
    # Mass 1 expected tof
    tof[:-1, min_stretch] = 1
    # Mass 4 expected tof
    tof[:-1, min_stretch * 2] = 1
    # Mass 9 expected tof
    tof[:-1, min_stretch * 3] = 1
    # Change the last TOF array to be shifted and 'stretched'
    # Mass 1 expected tof
    tof[-1, expected_stretch + expected_lag] = 1
    # Mass 4 expected tof
    tof[-1, expected_stretch * 2 + expected_lag] = 1
    # Mass 9 expected tof
    tof[-1, expected_stretch * 3 + expected_lag] = 1

    time = np.tile(np.arange(len(tof[0])), (15, 1))
    stretch, shift, mass_scale = time_to_mass(tof, time, masses)

    # Test with carbon mass
    carbon_time = (stretch[0] * np.sqrt(carbon_mass)) / 1e-6  # Convert ms to s
    mass = np.interp(carbon_time, time[0], mass_scale[0])
    assert np.allclose(carbon_mass, mass, rtol=1e-2)

    # Test shift is zero since peaks are aligned
    assert np.all(shift[:-1] == 0)
    # Test stretch factor matches expected 1400 ns in seconds
    assert np.all(stretch[:-1] == 1400 * 1e-9)
    # Test output shape
    assert mass_scale.shape == time.shape
    # Test that the last shift and stretch are the expected values
    assert shift[-1] == -expected_lag * idex_constants.FM_SAMPLING_RATE
    # Test stretch factor matches expected 1400 ns in seconds
    assert stretch[-1] == expected_stretch * 1e-9

    # Test with carbon mass
    carbon_time = (stretch[-1] * np.sqrt(carbon_mass) + shift[-1]) / 1e-6
    mass = np.interp(carbon_time, time[-1], mass_scale[-1])
    assert np.allclose(carbon_mass, mass, rtol=1e-2)


def test_time_to_mass_zero_correlation_warning(caplog):
    """
    Tests that the time_to_mass function correctly logs a warning if zero correlations
    are found between the TOF and expected mass times array.
    """
    masses = np.asarray([1, 4, 9])
    # Create a time of flight array that will result in no correlation between the
    # Expected tof peaks.
    tof = np.zeros((10, 8000))
    time = np.tile(np.arange(len(tof[0])), (10, 1))
    with caplog.at_level("WARNING"):
        time_to_mass(tof, time, masses)

    assert any(
        "There are no correlations found between the"
        " TOF array and the expected mass times array" in message
        for message in caplog.text.splitlines()
    )


def test_calculate_kappa():
    """Tests the functionality of calculate_kappa()."""
    # Create a 2d list of peak indices
    peaks = [[0, 1], [1, 2], [0, 1, 2]]

    # Create mass_scales array
    mass_scales = np.array(
        [
            [1.2, 2.2, 3.2],  # The kappa value for peaks 0,1 should be .2
            [1.4, 2.4, 3.4],  # The kappa value for peaks 1,2 should be .4
            [1.7, 2.7, 3.7],  # The kappa value for peaks 2,3,4 should be -0.3
        ]
    )
    kappas = calculate_kappa(mass_scales, peaks)

    assert np.allclose(list(kappas), [0.2, 0.4, -0.3], rtol=1e-12)


def test_calculate_snr():
    """Tests the functionality of calculate_snr()."""
    step = 0.5
    max_tof = 10
    time = np.arange(BaselineNoiseTime.START, 5, step)

    # Create a baseline noise array with an std of 1 and mean of 1
    baseline_noise = np.asarray([0, 0, 1, 2, 2])
    signal_length = len(time) - len(baseline_noise)
    tof_signal = np.full(int(signal_length), max_tof)

    tof = np.tile(np.append(baseline_noise, tof_signal), (3, 1))
    time = np.tile(time, (3, 1))

    snr = calculate_snr(tof, time)

    # Since std=1 and mean=1, SNR should be (max_tof - mean)/std
    assert np.all(snr == (max_tof - 1))


def test_calculate_snr_warning(caplog):
    """Tests that calculate_snr() throws warning if no baseline noise is found."""
    time = np.tile(np.arange(10), (3, 1))
    tof = np.ones_like(time)

    with caplog.at_level("WARNING"):
        calculate_snr(tof, time)
    assert any(
        "Unable to find baseline noise" in message
        for message in caplog.text.splitlines()
    )


def test_analyze_peaks_warning(caplog):
    """Tests that analyze_peaks() throws warning if the emg curve fit fails."""
    # Create a 2d list of peak indices
    peaks = [[2]]
    time = xr.DataArray(np.arange(6))
    # When there is a flat signal for TOF, we expect the fit to fail and a
    # warning to be logged.
    tof = np.ones_like(time)
    mass_scale = np.ones_like(time)
    with caplog.at_level("WARNING"):
        fit_params, area_under_curve, chisqr, redchi = analyze_peaks(
            tof, time, mass_scale, 0, peaks
        )
    assert any(
        "Failed to fit EMG curve" in message for message in caplog.text.splitlines()
    )
    # The fit_params, area_under_curve, chi square and reduced chi square arrays should
    # be zero
    np.testing.assert_array_equal(chisqr, np.zeros(chisqr.shape))
    np.testing.assert_array_equal(redchi, np.zeros(redchi.shape))
    np.testing.assert_array_equal(fit_params, np.zeros(fit_params.shape))
    np.testing.assert_array_equal(area_under_curve, np.zeros(area_under_curve.shape))


def test_load_calibration_files_returns_expected_t_rise_params(tmp_path):
    """Tests that t-rise ancillary values are loaded into t_rise_params."""
    expected_t_rise_params = np.array([1.27, -0.2, -2.1, -0.37, 5.3, 13.3, 13.3, 0.28])
    yield_values = np.array([0.06, 2.8, 5.9, 4.1, 13.0, 22.7, 8.2, 0.40, 1.47])

    t_rise_path = tmp_path / "t_rise.csv"
    yield_path = tmp_path / "yield.csv"
    _write_calibration_csv(t_rise_path, expected_t_rise_params)
    _write_calibration_csv(yield_path, yield_values)

    t_rise_params, _yield_params = load_calibration_files(
        {
            "l2a-calibration-curve-t-rise": t_rise_path,
            "l2a-calibration-curve-yield-params": yield_path,
        }
    )

    np.testing.assert_allclose(t_rise_params, expected_t_rise_params)


def test_load_calibration_files_returns_expected_yield_params(tmp_path):
    """Tests that yield ancillary values are loaded into yield_params."""
    t_rise_values = np.array([1.27, -0.2, -2.1, -0.37, 5.3, 13.3, 13.3, 0.28, 1.33])
    expected_yield_params = np.array([0.06, 2.8, 5.9, 4.1, 13.0, 22.7, 8.2, 0.40])

    t_rise_path = tmp_path / "t_rise.csv"
    yield_path = tmp_path / "yield.csv"
    _write_calibration_csv(t_rise_path, t_rise_values)
    _write_calibration_csv(yield_path, expected_yield_params)

    _t_rise_params, yield_params = load_calibration_files(
        {
            "l2a-calibration-curve-t-rise": t_rise_path,
            "l2a-calibration-curve-yield-params": yield_path,
        }
    )

    np.testing.assert_allclose(yield_params, expected_yield_params)


def test_log_smooth_powerlaw_yield_curve_at_10_km_s():
    """Tests that the yield calibration returns the expected value at 10 km/s."""
    yield_params = np.array([0.06, 2.8, 5.9, 4.1, 13.0, 22.7, 8.2, 0.40])

    log_yield = log_smooth_powerlaw(np.log10(10.0), yield_params[0], yield_params[1:])
    yield_value = 10**log_yield

    assert yield_value == pytest.approx(755.0, rel=1e-3)


def test_invert_rise_time_to_velocity_at_10_km_s():
    """Tests that the rise-time calibration can be inverted back to 10 km/s."""
    t_rise_params = np.array([1.27, -0.2, -2.1, -0.37, 5.3, 13.3, 13.3, 0.28])
    expected_velocity = 10.0
    t_rise = 10 ** log_smooth_powerlaw(
        np.log10(expected_velocity), float(t_rise_params[0]), t_rise_params[1:]
    )

    velocity_estimate = invert_rise_time_to_velocity(t_rise, t_rise_params)

    assert velocity_estimate == pytest.approx(expected_velocity, rel=1e-12)


def test_invert_rise_time_to_velocity_invalid_t_rise_returns_nan():
    """Tests that non-positive or non-finite rise times return NaN."""
    t_rise_params = np.array([1.27, -0.2, -2.1, -0.37, 5.3, 13.3, 13.3, 0.28])

    assert np.isnan(invert_rise_time_to_velocity(np.nan, t_rise_params))
    assert np.isnan(invert_rise_time_to_velocity(0.0, t_rise_params))


def test_calculate_velocity_and_mass_at_10_km_s():
    """Tests mass estimation using a mocked 10 km/s velocity solution."""
    t_rise_params = np.array([1.27, -0.2, -2.1, -0.37, 5.3, 13.3, 13.3, 0.28])
    yield_params = np.array([0.06, 2.8, 5.9, 4.1, 13.0, 22.7, 8.2, 0.40])
    sig_amp_pc = 10.0

    # This test intentionally bypasses the t_rise -> velocity inversion.
    # The t_rise calibration path is currently under review and will be
    # covered by a dedicated follow-up test once that behavior is finalized.
    mocked_root = mock.Mock()
    mocked_root.root = 1.0  # 10**1.0 == 10 km/s

    with mock.patch(
        "imap_processing.idex.idex_l2a.root_scalar", return_value=mocked_root
    ):
        velocity_estimate, mass_estimate = calculate_velocity_and_mass(
            sig_amp_pc, 2.0, t_rise_params, yield_params
        )

    expected_yield = 755.0090524738858
    expected_mass_kg = sig_amp_pc * 1e-12 / expected_yield

    assert velocity_estimate == pytest.approx(10.0, rel=1e-12)
    assert mass_estimate == pytest.approx(expected_mass_kg, rel=1e-12)


@pytest.mark.external_test_data
def test_velocity_and_mass_estimate(ancillary_files):
    """Tests that the velocity and mass estimate function."""
    # Load calibration coefficients from ancillary files
    t_rise_params = pd.read_csv(
        ancillary_files["l2a-calibration-curve-t-rise"], skiprows=1, header=None
    ).values.flatten()[:8]
    yield_params = pd.read_csv(
        ancillary_files["l2a-calibration-curve-yield-params"], skiprows=1, header=None
    ).values.flatten()[:8]
    expected_velocity = 5.0
    t_rise = 10 ** log_smooth_powerlaw(
        np.log10(expected_velocity), float(t_rise_params[0]), t_rise_params[1:]
    )
    estimates = calculate_velocity_and_mass(10, t_rise, t_rise_params, yield_params)
    assert len(estimates) == 2
    assert not np.any(np.isnan(estimates))
    assert estimates[0] == pytest.approx(expected_velocity, rel=1e-12)


def test_analyze_peaks_perfect_fits():
    """Tests that analyze_peaks() returns the expected fit params and areas."""
    event = 0
    # Create a 2d list of peak indices
    peak_1 = 7
    peak_2 = 25
    peak_3 = 80
    # Create tof array of ones
    time = xr.DataArray(np.arange(100))
    tof = np.zeros(100)
    mass_scale = np.arange(100) + 0.5
    # Only test peaks[0] this function is not vectorized but we pass in the full 2d peak
    # array.
    peaks = [np.asarray([peak_1, peak_2, peak_3]), np.asarray([])]
    sigma = 2.0
    lam = 1.0
    k = 1 / (lam * sigma)
    # Create a tof array with an emg curve at each peak
    for peak in peaks[event]:
        # Create a perfect emg curve
        mu = peak - 0.4
        gauss = exponnorm.pdf(time.data, k, mu, sigma)
        tof[peak - 5 : peak + 6] = gauss[peak - 5 : peak + 6]

    fit_params, area_under_curve, chisqr, redchi = analyze_peaks(
        tof, time, mass_scale, event, peaks
    )

    for peak in peaks[event]:
        mu = peak - 0.4
        mass = round(mass_scale[round(mu)])
        # Test that the fitted parameters at the mass index match our input parameters
        assert np.allclose(fit_params[mass], np.asarray([mu, sigma, lam]), rtol=1e-12)
        # Test that there is a value greater than zero at this index
        assert area_under_curve[mass] > 0
        # Test the goodness of fit
        assert np.all(chisqr < 1e-20)
        assert np.all(redchi < 1e-20)


def test_estimate_dust_mass_no_noise_removal():
    """
    Test that estimate_dust_mass() is fitting the signal properly when there is no
    noise removal.
    """
    # TODO: The IDEX team is iterating on this function and will provide more
    #  information soon.
    start_time = -60
    total_low_sampling_microseconds = 126.03  # see algorithm document.
    num_samples = 512

    # Create realistic low sampling time
    time = xr.DataArray(
        np.linspace(
            start_time, total_low_sampling_microseconds - start_time, num_samples
        )
    )
    signal = xr.DataArray(
        fit_impact(
            time.data,
            time_of_impact=0.0,
            constant_offset=1.0,
            amplitude=10.0,
            rise_time=0.371,
            discharge_time=0.371,
        )
    )
    param, sig_amp, chisqr, redchi, result = estimate_dust_mass(
        time, signal, remove_noise=False
    )
    # Assert that the chi square value indicates a very good fit
    assert chisqr <= 1e-12

    assert np.allclose(result, signal)


def test_estimate_dust_mass_logs_baseline_warning(caplog):
    """
    Test that estimate_dust_mass() logs a warning if no baseline is found.
    """
    time = xr.DataArray(np.linspace(-60, 60, 16))
    signal = xr.DataArray(np.linspace(0, 1, 16))
    original_any = np.any

    def fake_any(arr):
        fake_any.calls += 1
        if fake_any.calls == 1:
            return False
        return original_any(arr)

    fake_any.calls = 0

    with (
        mock.patch("imap_processing.idex.idex_l2a.np.any", side_effect=fake_any),
        mock.patch(
            "imap_processing.idex.idex_l2a.curve_fit",
            return_value=(np.array([0.0, 0.0, 1.0, 0.371, 37.1]), None),
        ),
        caplog.at_level("WARNING"),
    ):
        estimate_dust_mass(time, signal)

    assert any(
        "Unable to find baseline noise" in message
        for message in caplog.text.splitlines()
    )


def test_estimate_dust_mass_remove_noise_logs_debug(caplog):
    """
    Test that estimate_dust_mass() logs that remove_noise is ignored.
    """
    time = xr.DataArray(np.linspace(-60, 60, 64))
    signal = xr.DataArray(
        fit_impact(
            time.data,
            time_of_impact=0.0,
            constant_offset=1.0,
            amplitude=10.0,
            rise_time=0.371,
            discharge_time=0.371,
        )
    )

    with caplog.at_level("DEBUG"):
        estimate_dust_mass(time, signal, remove_noise=True)

    assert any(
        "remove_noise is ignored for this fit path" in message
        for message in caplog.text.splitlines()
    )


def test_estimate_dust_mass_nonfinite_signal_fallbacks():
    """
    Test fallback handling when the input signal is entirely non-finite.
    """
    time = xr.DataArray(np.linspace(-60, 60, 16))
    signal = xr.DataArray(np.full(16, np.nan))

    with mock.patch(
        "imap_processing.idex.idex_l2a.curve_fit",
        return_value=(np.array([0.0, 0.0, 1.0, 0.371, 37.1]), None),
    ) as mocked_curve_fit:
        estimate_dust_mass(time, signal)

    assert np.isnan(mocked_curve_fit.call_args.kwargs["p0"][2])


def test_estimate_dust_mass_non_ion_grid_negative_amplitude_fallback():
    """
    Test the non-Ion_Grid negative-amplitude fallback path.
    """
    time = xr.DataArray(np.linspace(-60, 60, 16))
    signal = xr.DataArray(np.full(16, -2.0))

    with mock.patch(
        "imap_processing.idex.idex_l2a.curve_fit",
        return_value=(np.array([0.0, -2.0, -2.0, 0.371, 37.1]), None),
    ) as mocked_curve_fit:
        estimate_dust_mass(time, signal)

    assert mocked_curve_fit.call_args.kwargs["p0"][2] == -2.0


def test_estimate_dust_mass_ion_grid_negative_amplitude_bounds():
    """
    Test that Ion Grid fits allow negative amplitudes.
    """
    time = xr.DataArray(np.linspace(-60, 60, 64))
    signal = xr.DataArray(
        fit_impact(
            time.data,
            time_of_impact=0.0,
            constant_offset=0.0,
            amplitude=-5.0,
            rise_time=0.371,
            discharge_time=0.371,
        )
    )

    with mock.patch(
        "imap_processing.idex.idex_l2a.curve_fit",
        return_value=(np.array([0.0, 0.0, -5.0, 0.371, 37.1]), None),
    ) as mocked_curve_fit:
        estimate_dust_mass(time, signal, waveform_name="Ion_Grid")

    bounds = mocked_curve_fit.call_args.kwargs["bounds"]
    assert bounds[0][2] == -np.inf
    assert bounds[1][2] < 0.0


def test_estimate_dust_mass_curve_fit_failure_returns_nans(caplog):
    """
    Test that estimate_dust_mass() returns NaNs if the fit fails.
    """
    time = xr.DataArray(np.linspace(-60, 60, 16))
    signal = xr.DataArray(np.linspace(0, 1, 16))

    with (
        mock.patch(
            "imap_processing.idex.idex_l2a.curve_fit",
            side_effect=RuntimeError("fit failed"),
        ),
        caplog.at_level("WARNING"),
    ):
        param, sig_amp, chisqr, redchi, result = estimate_dust_mass(time, signal)

    assert any("Failed to fit curve" in message for message in caplog.text.splitlines())
    assert np.all(np.isnan(param))
    assert np.isnan(sig_amp)
    assert np.isnan(chisqr)
    assert np.isnan(redchi)
    assert np.all(np.isnan(result))


def test_lowpass_filter():
    """
    Tests that the lowpass filter is filtering out high frequency signals.

    Look at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
    for the source of the testing example.
    """

    time = np.linspace(-60, 60, 512)
    # Calculate nyquist frequency to help get cutoff.
    # This is the highest frequency that can be captured
    time_between_samples = time[1] - time[0]
    nqf = (1 / time_between_samples) / 2
    # Choose cutoff of 0.125 times the Nyquist frequency
    cutoff = nqf * 0.125
    # Create two signals with different frequencies and combine them
    low_freq = cutoff / 4  # Lower than cutoff
    high_freq = nqf  # The nyquist frequency is much higher than the cutoff
    # Create sine signals
    signal_low = np.sin(2 * np.pi * low_freq * time)
    signal_high = np.sin(2 * np.pi * high_freq * time)
    combined_sig = signal_low + signal_high
    # The filter should filter out the high frequency signal
    filtered_sig = butter_lowpass_filter(time, combined_sig, cutoff)
    # Assert that the filtered signal is relatively close to the original low
    # frequency signal.
    np.allclose(filtered_sig, signal_low)


def test_remove_signal_noise():
    """
    Tests that remove_signal_noise() function is filtering out sine wave and linear
    noise due to "microphonics"
    """
    start_time = -60
    total_low_sampling_microseconds = 126.03  # see algorithm document.
    num_samples = 512

    # Create realistic low sampling time
    time = np.linspace(
        start_time, total_low_sampling_microseconds - start_time, num_samples
    )

    mask = time <= (start_time + total_low_sampling_microseconds) / 2
    noisy_signal = mock_microphonics_noise(time)
    # Filter signal
    filtered_sig = remove_signal_noise(time, noisy_signal, mask)

    np.allclose(filtered_sig, np.zeros_like(filtered_sig), atol=1e-2)


def test_remove_signal_noise_no_sine_wave(caplog):
    """
    Tests that remove_signal_noise() function filters linear noise when there is no
    sine wave.
    """
    time = np.linspace(-60, 60, 512)
    # linear signal to create noise
    signal = time * 10
    mask = time <= 0.5
    # Filter signal
    filtered_sig = remove_signal_noise(time, signal, mask)
    # Test that the filtered signal is close to zero
    assert np.allclose(filtered_sig, np.zeros_like(filtered_sig), rtol=1e-24)


def test_chi_square():
    """
    Test that chi_square() function calculates the expected values for an array of given
    residuals.
    """
    residual = 3
    nparams = 2
    exp = np.array([16, 18, 16, 14, 12, 12])
    obs = exp + residual

    expected_chi_square = np.square(residual) * len(exp)
    expected_red_chi_square = expected_chi_square / (len(exp) - nparams)

    chisqr, redchi = chi_square(obs, exp, nparams)

    assert chisqr == expected_chi_square
    assert redchi == expected_red_chi_square
