"""Tests the L2a processing for IDEX data"""

import numpy as np
import pytest
import xarray as xr
from scipy.stats import exponnorm

from imap_processing.idex import idex_constants
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.idex.idex_l2a import (
    BaselineNoiseTime,
    analyze_peaks,
    calculate_kappa,
    calculate_snr,
    emg,
    estimate_dust_mass,
    fit_impact,
    idex_l2a,
    time_to_mass,
)


@pytest.fixture(scope="module")
def l2a_dataset(decom_test_data: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
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
    expected_src = "imap_idex_l2a_sci"
    assert l2a_dataset.attrs["Logical_source"] == expected_src


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


@pytest.mark.filterwarnings("ignore:invalid")
@pytest.mark.filterwarnings("ignore:overflow")
def test_analyze_peaks_warning(caplog):
    """Tests that analyze_peaks() throws warning if the emg curve fit fails."""
    # Create a 2d list of peak indices
    peaks = [[2, 3, 4]]
    time = xr.DataArray(np.arange(10))
    # When there is a flat signal for TOF, we expect the fit to fail and a
    # warning to be logged.
    tof = np.ones_like(time)
    with caplog.at_level("WARNING"):
        fit_params, area_under_curve = analyze_peaks(tof, time, 0, peaks)
    assert any(
        "Failed to fit EMG curve" in message for message in caplog.text.splitlines()
    )

    # The fit_params and area_under_curve arrays should be zero
    assert np.all(fit_params == 0)
    assert np.all(area_under_curve == 0)


def test_emg():
    """Tests that emg() calculates an expected exponentially modified gaussian"""
    mu = 4
    sigma = 2.0
    lam = 1.0
    time = xr.DataArray(np.arange(100))
    # The scipy.stats.exponnorm function's location and scale parameters match directly
    # with mu and sigma from the standard EMG function, while its shape parameter K is
    # found by taking one divided by the product of sigma and lambda.
    # see:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html
    # for more details
    k = 1 / (lam * sigma)
    # Calculate the EMGs
    g1 = exponnorm.pdf(time, k, loc=mu, scale=sigma)
    g2 = emg(time, mu, sigma, lam)

    assert np.allclose(g1, g2)


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
    # Only test peaks[0] this function is not vectorized but we pass in the full 2d peak
    # array.
    peaks = [np.asarray([peak_1, peak_2, peak_3]), np.asarray([])]
    sigma = 2.0
    lam = 1.0
    # Create a tof array with an emg curve at each peak
    for peak in peaks[event]:
        # Create a perfect emg curve
        mu = peak - 0.4
        gauss = emg(time.data, mu, sigma, lam)
        tof[peak - 5 : peak + 6] = gauss[peak - 5 : peak + 6]

    fit_params, area_under_curve = analyze_peaks(tof, time, event, peaks)

    for peak in peaks[event]:
        mu = peak - 0.4
        idx = round(mu)
        # Test that the fitted parameters at the mass index match our input parameters
        assert np.allclose(fit_params[idx], np.asarray([mu, sigma, lam]), rtol=1e-12)
        # Test that there is a value greater than zero at this index
        assert area_under_curve[idx] > 0


def test_estimate_dust_mass_no_noise_removal():
    """
    Test that estimate_dust_mass() is fitting the signal properly when there is no
    noise removal.
    """
    pass
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
