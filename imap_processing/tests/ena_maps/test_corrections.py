"""Test coverage for ena_maps.corrections module."""

from unittest import mock

import numpy as np
import pytest

from imap_processing.ena_maps.utils.corrections import PowerLawFluxCorrector


@pytest.fixture
def hi_coeffs_file(ena_maps_test_data_path):
    """Define the location of the hi coefficients file."""
    return (
        ena_maps_test_data_path
        / "imap_hi_90sensor-esa-eta-fit-factors_20240101_v001.csv"
    )


@pytest.fixture
def lo_coeffs_file(ena_maps_test_data_path):
    """Define the location of the hi coefficients file."""
    return ena_maps_test_data_path / "imap_lo_esa-eta-fit-factors_20240101_v001.csv"


class TestPowerLawFluxCorrector:
    """Test suite for ena_maps.corrections.PowerLawFluxCorrector."""

    def test_load_coefficients_invalid_file(self, tmp_path):
        """Test that loading a missing CSV file raises FileNotFoundError."""

        with pytest.raises(FileNotFoundError):
            PowerLawFluxCorrector(tmp_path / "missing.csv")

    def test_eta_esa_non_negative(self, lo_coeffs_file):
        """Test eta_esa will not return negative values."""

        corr = PowerLawFluxCorrector(lo_coeffs_file)
        k = np.array([1, 2])
        # Experimentally found that gamma=10 produces negative eta
        gamma = np.array([1.5, 10])
        eta = corr.eta_esa(k, gamma)
        assert eta[1] == 1

    def test_estimate_power_law_with_uncertainties(self):
        """Test slope estimation with flux uncertainties."""

        fluxes = np.array([10, 20, 40, 80, 160, 320, 640])
        energies = np.arange(8) + 1
        uncertainties = np.sqrt(fluxes)
        gamma, delta_gamma = PowerLawFluxCorrector.estimate_power_law_slope(
            fluxes, energies, uncertainties
        )
        assert np.all(np.isfinite(gamma))
        assert delta_gamma is not None
        assert np.all(delta_gamma > 0)

    def test_estimate_power_law_with_zero_flux(self):
        """Test slope estimation falls back to linear differencing."""

        fluxes = np.array([10, 0, 40, 60, 0, 0, 80])
        uncertainties = np.maximum(0.1 * fluxes, 1)
        expected_gamma = np.array(
            [
                0,  # End point should fail to find slope
                np.log(40 / 10)
                / np.log(3 / 1),  # Normal central differencing log-slope
                np.log(60 / 40)
                / np.log(4 / 3),  # Fallback to forward linear differencing
                np.log(60 / 40)
                / np.log(4 / 3),  # Fallback to backward linear differencing
                0,  # No differencing scheme works
                0,  # No differencing scheme works
                0,  # End point fails to find slope
            ]
        )
        expected_delta_gamma = np.array(
            [
                0,
                np.sqrt(2 * (0.1**2)) / np.log(3 / 1),
                np.sqrt(2 * (0.1**2)) / np.log(4 / 3),
                np.sqrt(2 * (0.1**2)) / np.log(4 / 3),
                0,
                0,
                0,
            ]
        )
        energies = np.arange(len(fluxes)) + 1
        corr = PowerLawFluxCorrector
        gamma, delta_gamma = corr.estimate_power_law_slope(
            fluxes, energies, uncertainties
        )
        np.testing.assert_array_almost_equal(gamma, expected_gamma)
        np.testing.assert_array_almost_equal(delta_gamma, expected_delta_gamma)

    def test_predictor_corrector_nonconvergence(self, lo_coeffs_file):
        """Test predictor-corrector stops after max_iterations."""

        corr = PowerLawFluxCorrector(lo_coeffs_file)
        fluxes = (np.arange(7) * 1000**2)[::-1]
        energies = np.arange(1, 8) + 1
        _, _, n_iter = corr.predictor_corrector_iteration(
            fluxes,
            np.sqrt(fluxes),
            energies,
            max_iterations=3,
            convergence_threshold=1e-12,
        )
        assert n_iter == 3

    def create_lo_test_data(self):
        """Create synthetic Lo data to test."""
        # Test data matches data from MappingValidation_transforms_V02.xlsx
        # Example data - 7 energy levels
        energies = np.array([16.35, 30.56, 56.42, 105.21, 199.79, 407.49, 795.28])  # eV

        # Example observed fluxes
        observed_fluxes = np.array([1000, 800, 50, 200, 1, 30, 10])
        delta_fluxes = np.sqrt(observed_fluxes)  # Poisson uncertainties
        sigma_fluxes = 0.1 * observed_fluxes  # 10% systematic uncertainties

        # Example background fluxes (much smaller than signal)
        background_fluxes = 0.01 * observed_fluxes
        delta_background = np.sqrt(background_fluxes)
        sigma_background = 0.15 * background_fluxes

        flux_dict = {
            "J": observed_fluxes,
            "delta_J": delta_fluxes,
            "sigma_J": sigma_fluxes,
        }

        background_dict = {
            "J_B": background_fluxes,
            "delta_J_B": delta_background,
            "sigma_J_B": sigma_background,
        }

        return energies, flux_dict, background_dict

    def create_hi_test_data(self):
        """Create synthetic Hi data to test."""
        # Test data matches data from MappingValidation_Hi_transforms_V03.xlsx
        # Example data - 9 energy levels
        energies = (
            np.array([0.5, 0.75, 1.1, 1.65, 2.5, 3.75, 5.7, 8.52, 12.80]) * 1000
        )  # eV

        # Example observed fluxes
        observed_fluxes = np.array([1000, 800, 50, 200, 1, 30, 10, 2, 5])
        delta_fluxes = np.sqrt(observed_fluxes)  # Poisson uncertainties
        sigma_fluxes = 0.1 * observed_fluxes  # 10% systematic uncertainties

        # Example background fluxes (much smaller than signal)
        background_fluxes = 0.01 * observed_fluxes
        delta_background = np.sqrt(background_fluxes)
        sigma_background = 0.15 * background_fluxes

        flux_dict = {
            "J": observed_fluxes,
            "delta_J": delta_fluxes,
            "sigma_J": sigma_fluxes,
        }

        background_dict = {
            "J_B": background_fluxes,
            "delta_J_B": delta_background,
            "sigma_J_B": sigma_background,
        }

        return energies, flux_dict, background_dict

    def test_predictor_corrector_lo_example(self, lo_coeffs_file):
        """Test correction using sample data from Nathan's spreadsheet."""
        flux_corr = PowerLawFluxCorrector(lo_coeffs_file)
        energies, flux_dict, background_dict = self.create_lo_test_data()
        corrected_fluxes, corrected_unc, _ = flux_corr.predictor_corrector_iteration(
            flux_dict["J"], flux_dict["delta_J"], energies
        )
        expected_corr_fluxes = np.array(
            [
                926.9339867,
                553.5811764,
                44.32189088,
                118.6296225,
                0.911160458,
                29.3853061,
                7.828285642,
            ]
        )
        np.testing.assert_allclose(corrected_fluxes, expected_corr_fluxes, rtol=1e-2)

    def test_predictor_corrector_hi_example(self, hi_coeffs_file):
        """Test correction using sample data from Nathan's spreadsheet."""
        flux_corr = PowerLawFluxCorrector(hi_coeffs_file)
        energies, flux_dict, background_dict = self.create_hi_test_data()
        corrected_fluxes, corrected_unc, _ = flux_corr.predictor_corrector_iteration(
            flux_dict["J"], flux_dict["delta_J"], energies
        )
        expected_corr_fluxes = np.array(
            [
                934.9348044,
                528.302229,
                44.47463759,
                111.0485641,
                0.915876546,
                27.96414141,
                7.587531207,
                1.96618265,
                4.782030232,
            ]
        )
        np.testing.assert_allclose(corrected_fluxes, expected_corr_fluxes, rtol=1e-2)

    @mock.patch(
        "imap_processing.ena_maps.utils.corrections.PowerLawFluxCorrector.predictor_corrector_iteration"
    )
    def test_apply_flux_correction(self, mock_predictor_corrector, hi_coeffs_file):
        """Test applying the correction to map data."""
        mock_predictor_corrector.side_effect = lambda f, d_f, e: (f * 2, d_f / 2, 0)
        flux = np.arange(90).reshape(9, 10)
        delta_flux = np.sqrt(flux)
        energies = np.arange(flux.shape[0])

        flux_corr = PowerLawFluxCorrector(hi_coeffs_file)
        corrected_flux, corrected_delta_flux = flux_corr.apply_flux_correction(
            flux, delta_flux, energies
        )
        np.testing.assert_array_equal(corrected_flux, flux * 2)
        np.testing.assert_array_equal(corrected_delta_flux, delta_flux / 2)
