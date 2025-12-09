import astropy_healpix.healpy as hp
import numpy as np
import pytest

from imap_processing.ultra.l1c.l1c_lookup_utils import (
    get_scattering_thresholds_for_energy,
    get_spacecraft_pointing_lookup_tables,
    get_static_deadtime_ratios,
    mask_below_fwhm_scattering_threshold,
)


@pytest.mark.external_test_data
def test_get_spacecraft_pointing_lookup_tables(ancillary_files):
    """Test get_spacecraft_pointing_lookup_tables."""
    instrument_id = 90
    (
        for_indices_by_spin_phase,
        theta_vals,
        phi_vals,
        ra_and_dec,
        boundary_scale_factors,
    ) = get_spacecraft_pointing_lookup_tables(ancillary_files, instrument_id)
    npix = hp.nside2npix(32) - 1  # test files dont have headers
    # Test shapes
    # There should be 498 spin phase steps. In the real test files there will be 15000
    cols = 498
    assert for_indices_by_spin_phase.shape == (cols, npix)
    assert theta_vals.shape == (cols, npix)
    assert phi_vals.shape == (cols, npix)
    assert ra_and_dec.shape == (2, npix)

    # Value tests
    assert for_indices_by_spin_phase.dtype == bool
    assert theta_vals.dtype == np.float64
    assert phi_vals.dtype == np.float64
    assert ra_and_dec.dtype == np.float64
    assert boundary_scale_factors.dtype == np.float64


@pytest.mark.external_test_data
def test_get_mask_below_fwhm_scattering_threshold(ancillary_files):
    """Tests function get_mask_below_fwhm_scattering_threshold."""
    energy = np.array([5])  # At energy 5, the FWHM threshold is 10
    thresholds = get_scattering_thresholds_for_energy(energy, ancillary_files)
    theta_coeffs = np.array(
        [
            [np.nan, 10],  # This will result in a NaN value (False)
            [5, -0.1],  # FWHM value below the threshold (True)
            [4, -0.1],  # FWHM value below the threshold (True)
        ]
    )
    phi_coeffs = np.array(
        [
            [3, -0.1],  # FWHM value below the threshold (True)
            [5, -0.1],  # FWHM value below the threshold (True)
            [15, -0.1],  # FWHM value above the threshold (False)
        ]
    )
    # Only indices where both the theta and phi coefficients are below the FWHM
    # threshold should be True.
    expected_pixel_mask = np.array([[False], [True], [False]])
    pixel_mask, scat_theta, scat_phi = mask_below_fwhm_scattering_threshold(
        theta_coeffs, phi_coeffs, energy[np.newaxis, :], thresholds
    )
    np.testing.assert_array_equal(pixel_mask.shape, (3, 1))
    np.testing.assert_array_equal(pixel_mask, expected_pixel_mask)


@pytest.mark.external_test_data
def test_get_mask_below_fwhm_scattering_threshold_zero(ancillary_files):
    """Tests function get_mask_below_fwhm_scattering_threshold."""
    energy = np.array([0])  # At energy 0, the FWHM threshold is 0
    thresholds = get_scattering_thresholds_for_energy(energy, ancillary_files)
    theta_coeffs = np.array(
        [
            [np.nan, 10],  # This will result in a NaN value (False)
            [5, -0.1],  # FWHM value below the threshold (True)
            [4, -0.1],  # FWHM value below the threshold (True)
        ]
    )
    phi_coeffs = np.array(
        [
            [3, -0.1],  # FWHM value below the threshold (True)
            [5, -0.1],  # FWHM value below the threshold (True)
            [15, -0.1],  # FWHM value above the threshold (False)
        ]
    )
    # Since energy is zero, all should be False although some pixels are below threshold
    expected_pixel_mask = np.array([[False], [False], [False]])
    pixel_mask, scat_theta, scat_phi = mask_below_fwhm_scattering_threshold(
        theta_coeffs, phi_coeffs, energy, thresholds
    )
    np.testing.assert_array_equal(pixel_mask.shape, (3, 1))
    np.testing.assert_array_equal(pixel_mask, expected_pixel_mask)


@pytest.mark.external_test_data
def test_get_static_deadtime_ratios(ancillary_files):
    """Test get_static_deadtime_ratios function."""
    # test 45
    spin_phase, dt_ratio = get_static_deadtime_ratios(45, ancillary_files)
    # Test shape
    # TODO confirm if the duplicate row in the 45 LUT is a mistake
    np.testing.assert_array_equal(dt_ratio.shape, (720,))
    # Test values
    assert np.all((dt_ratio >= 0.0) & (dt_ratio <= 1.0))
    # test 90
    spin_phase, dt_ratio = get_static_deadtime_ratios(90, ancillary_files)
    # Test shape
    np.testing.assert_array_equal(dt_ratio.shape, (721,))
    # Test the values
    assert np.all((dt_ratio >= 0.0) & (dt_ratio <= 1.0))
