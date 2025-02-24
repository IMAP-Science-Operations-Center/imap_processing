"""Tests the L2a processing for IDEX data"""

import numpy as np
import pytest
import xarray as xr

from imap_processing.idex import idex_constants
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.idex.idex_l2a import (
    BaselineNoiseTime,
    calculate_kappa,
    calculate_snr,
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
