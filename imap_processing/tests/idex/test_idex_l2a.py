"""Tests the L2a processing for IDEX data"""

from unittest import mock

import numpy as np
import pytest
import xarray as xr
from scipy.stats import exponnorm

from imap_processing.idex import idex_constants
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.idex.idex_l2a import (
    BaselineNoiseTime,
    analyze_peaks,
    butter_lowpass_filter,
    calculate_kappa,
    calculate_snr,
    estimate_dust_mass,
    fit_impact,
    idex_l2a,
    remove_signal_noise,
    time_to_mass,
)


def mock_microphonics_noise(time: np.ndarray) -> np.ndarray:
    """Function to mock signal noise (linear and sine wave) due to microphonics."""
    noise_frequency = idex_constants.TARGET_NOISE_FREQUENCY
    phase_shift = 45
    amp = 10
    # Create a sine wave signal
    sine_signal = amp * np.sin(2 * np.pi * noise_frequency * time + phase_shift)
    # Combine the sine wave signals with a linear signal to create noise
    combined_sig = sine_signal + (time * 5)

    return combined_sig


@pytest.fixture(scope="module")
def l2a_dataset(decom_test_data: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    with mock.patch("imap_processing.idex.idex_l1b.get_spice_data", return_value={}):
        dataset = idex_l2a(
            idex_l1b(decom_test_data, data_version="001"), data_version="001"
        )
    return dataset


def test_l2a_cdf_filenames(l2a_dataset: xr.Dataset):
    """Tests that the ``idex_l2a`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    l2a_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_src = "imap_idex_l2a_sci-1week"
    assert l2a_dataset.attrs["Logical_source"] == expected_src


def test_l2a_cdf_variables(l2a_dataset: xr.Dataset):
    """Tests that the ``idex_l2a`` function generates datasets
    with the expected variables.

    Parameters
    ----------
    l2a_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_vars = [
        "mass",
        "target_low_fit_parameters",
        "target_low_fit_imapct_charge",
        "target_low_fit_imapct_mass_estimate",
        "target_low_chi_squared",
        "target_low_reduced_chi_squared",
        "target_low_fit_results",
        "target_high_fit_parameters",
        "target_high_fit_imapct_charge",
        "target_high_fit_imapct_mass_estimate",
        "target_high_chi_squared",
        "target_high_reduced_chi_squared",
        "target_high_fit_results",
        "ion_grid_fit_parameters",
        "ion_grid_fit_imapct_charge",
        "ion_grid_fit_imapct_mass_estimate",
        "ion_grid_chi_squared",
        "ion_grid_reduced_chi_squared",
        "ion_grid_fit_results",
    ]

    cdf_vars = l2a_dataset.variables
    for var in expected_vars:
        assert var in cdf_vars


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
        fit_params, area_under_curve = analyze_peaks(tof, time, mass_scale, 0, peaks)
    assert any(
        "Failed to fit EMG curve" in message for message in caplog.text.splitlines()
    )

    # The fit_params and area_under_curve arrays should be zero
    assert np.all(fit_params == 0)
    assert np.all(area_under_curve == 0)


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

    fit_params, area_under_curve = analyze_peaks(tof, time, mass_scale, event, peaks)

    for peak in peaks[event]:
        mu = peak - 0.4
        mass = round(mass_scale[round(mu)])
        # Test that the fitted parameters at the mass index match our input parameters
        assert np.allclose(fit_params[mass], np.asarray([mu, sigma, lam]), rtol=1e-12)
        # Test that there is a value greater than zero at this index
        assert area_under_curve[mass] > 0


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
