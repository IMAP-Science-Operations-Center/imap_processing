"""Test coverage for ena_maps.corrections module."""

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps.ena_maps import HiPointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.corrections import (
    PowerLawFluxCorrector,
    _add_cartesian_look_direction,
    _calculate_compton_getting_transform,
    add_spacecraft_position_and_velocity_to_pset,
    apply_compton_getting_correction,
    calculate_ram_mask,
    get_pset_directional_mask,
    interpolate_map_flux_to_helio_frame,
)
from imap_processing.spice import geometry


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

        # Create 2D arrays (n_energy, n_pixels)
        fluxes = np.array([10, 20, 40, 80, 160, 320, 640])[:, np.newaxis]
        energies = np.arange(7) + 1
        uncertainties = np.sqrt(fluxes)
        gamma, delta_gamma = PowerLawFluxCorrector.estimate_power_law_slope(
            fluxes, energies, uncertainties
        )
        assert np.all(np.isfinite(gamma))
        assert delta_gamma is not None
        assert np.all(delta_gamma > 0)

    def test_estimate_power_law_with_zero_flux(self):
        """Test slope estimation falls back to linear differencing."""

        # Create 2D arrays (n_energy, n_pixels)
        fluxes = np.array([10, 0, 40, 60, 0, 0, 80])[:, np.newaxis]
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
        )[:, np.newaxis]
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
        )[:, np.newaxis]
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
        # Create 2D arrays (n_energy, n_pixels)
        fluxes = ((np.ones(7) * 1000**2)[::-1])[:, np.newaxis]
        energies = np.arange(1, 8) + 1
        _, _, n_iter = corr.predictor_corrector_iteration(
            fluxes,
            np.sqrt(fluxes),
            energies,
            max_iterations=3,
            convergence_threshold=1e-12,
        )
        assert np.all(n_iter == 3)

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
        # Reshape to 2D arrays (n_energy, n_pixels)
        corrected_fluxes, corrected_unc, _ = flux_corr.predictor_corrector_iteration(
            flux_dict["J"][:, np.newaxis], flux_dict["delta_J"][:, np.newaxis], energies
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
        np.testing.assert_allclose(
            corrected_fluxes.squeeze(), expected_corr_fluxes, rtol=1e-2
        )

    def test_predictor_corrector_hi_example(self, hi_coeffs_file):
        """Test correction using sample data from Nathan's spreadsheet."""
        flux_corr = PowerLawFluxCorrector(hi_coeffs_file)
        energies, flux_dict, background_dict = self.create_hi_test_data()
        # Reshape to 2D arrays (n_energy, n_pixels)
        corrected_fluxes, corrected_unc, _ = flux_corr.predictor_corrector_iteration(
            flux_dict["J"][:, np.newaxis], flux_dict["delta_J"][:, np.newaxis], energies
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
        np.testing.assert_allclose(
            corrected_fluxes.squeeze(), expected_corr_fluxes, rtol=1e-2
        )

    def test_predictor_corrector_zero_flux_convergence(self, hi_coeffs_file):
        """Test that convergence is achieved when we have a zero flux."""
        flux_corr = PowerLawFluxCorrector(hi_coeffs_file)
        energies, flux_dict, background_dict = self.create_hi_test_data()
        # set flux for ESA 9 to zero
        flux_dict["J"][-1] = 0
        # Reshape to 2D arrays (n_energy, n_pixels)
        _, _, n_iterations = flux_corr.predictor_corrector_iteration(
            flux_dict["J"][:, np.newaxis], flux_dict["delta_J"][:, np.newaxis], energies
        )
        assert np.all(n_iterations < 20)

    @mock.patch(
        "imap_processing.ena_maps.utils.corrections.PowerLawFluxCorrector.predictor_corrector_iteration"
    )
    def test_apply_flux_correction(self, mock_predictor_corrector, hi_coeffs_file):
        """Test applying the correction to map data."""
        # Mock returns 2D arrays (n_energy, n_pixels) and n_iterations
        mock_predictor_corrector.side_effect = lambda f, d_f, e: (
            f * 2,
            d_f / 2,
            np.zeros(f.shape[1], dtype=int),
        )

        # Create xarray DataArrays with energy dimension
        flux_data = np.arange(90).reshape(9, 10)
        delta_flux_data = np.sqrt(flux_data)
        energies_data = np.arange(flux_data.shape[0])

        flux = xr.DataArray(
            flux_data,
            dims=["energy", "spatial"],
            coords={"energy": energies_data, "spatial": np.arange(10)},
        )
        delta_flux = xr.DataArray(
            delta_flux_data,
            dims=["energy", "spatial"],
            coords={"energy": energies_data, "spatial": np.arange(10)},
        )
        energies = xr.DataArray(
            energies_data, dims=["energy"], coords={"energy": energies_data}
        )

        flux_corr = PowerLawFluxCorrector(hi_coeffs_file)
        corrected_flux, corrected_delta_flux = flux_corr.apply_flux_correction(
            flux, delta_flux, energies
        )

        # Verify output is xarray DataArray with correct dimensions
        assert isinstance(corrected_flux, xr.DataArray)
        assert isinstance(corrected_delta_flux, xr.DataArray)
        assert corrected_flux.dims == flux.dims
        assert corrected_delta_flux.dims == delta_flux.dims

        # Verify values are correct
        np.testing.assert_array_equal(corrected_flux.values, flux_data * 2)
        np.testing.assert_array_equal(corrected_delta_flux.values, delta_flux_data / 2)

    def test_estimate_power_law_slope_multi_pixel(self):
        """Test slope estimation with multiple spatial pixels (true 2D array)."""
        # Create test data with 7 energy levels and 12 spatial pixels
        # Shape: (n_energy=7, n_pixels=12)
        n_energy = 7
        n_pixels = 12

        # Create varying flux values across pixels
        energies = np.arange(n_energy) + 1
        base_fluxes = np.array([10, 20, 40, 80, 160, 320, 640])

        # Create 2D array where each pixel has slightly different flux scaling
        fluxes = (
            base_fluxes[:, np.newaxis] * np.linspace(0.8, 1.2, n_pixels)[np.newaxis, :]
        )
        uncertainties = np.sqrt(fluxes)

        gamma, delta_gamma = PowerLawFluxCorrector.estimate_power_law_slope(
            fluxes, energies, uncertainties
        )

        # Check output shapes
        assert gamma.shape == (n_energy, n_pixels)
        assert delta_gamma.shape == (n_energy, n_pixels)

        # All slopes should be finite and non-zero for this simple power-law data
        assert np.all(np.isfinite(gamma))
        assert np.all(np.isfinite(delta_gamma))

        # All slopes should be positive (fluxes increase with energy)
        assert np.all(gamma > 0)
        assert np.all(delta_gamma > 0)

    def test_estimate_power_law_slope_with_zeros_multi_pixel(self):
        """Test slope estimation with zero fluxes in multi-pixel array."""
        # Create test data with some zero fluxes at different locations per pixel
        n_energy = 7
        n_pixels = 5

        energies = np.arange(n_energy) + 1

        # Create different zero patterns for each pixel
        fluxes = np.array(
            [
                [10, 0, 10, 10, 10],  # energy 0: zero at pixel 1
                [20, 20, 0, 20, 20],  # energy 1: zero at pixel 2
                [40, 40, 40, 0, 40],  # energy 2: zero at pixel 3
                [60, 60, 60, 60, 0],  # energy 3: zero at pixel 4
                [0, 0, 0, 0, 80],  # energy 4: zeros at pixels 0-3
                [0, 80, 80, 80, 80],  # energy 5: zero at pixel 0
                [80, 80, 80, 80, 80],  # energy 6: no zeros
            ]
        )
        uncertainties = np.maximum(0.1 * fluxes, 1)

        gamma, delta_gamma = PowerLawFluxCorrector.estimate_power_law_slope(
            fluxes, energies, uncertainties
        )

        # Check output shapes
        assert gamma.shape == (n_energy, n_pixels)
        assert delta_gamma.shape == (n_energy, n_pixels)

        # Where we have valid data on both sides, slopes should be non-zero
        # Last energy level with all valid fluxes should have positive slopes
        assert np.all(gamma[-1, :] >= 0)

    def test_predictor_corrector_multi_pixel(self, lo_coeffs_file):
        """Test predictor-corrector with multiple spatial pixels."""
        corr = PowerLawFluxCorrector(lo_coeffs_file)

        # Create 2D array with 7 energy levels and 8 spatial pixels
        energies = np.array([16.35, 30.56, 56.42, 105.21, 199.79, 407.49, 795.28])
        n_energy = len(energies)
        n_pixels = 8

        # Create base fluxes that vary across pixels
        base_fluxes = np.array([1000, 800, 50, 200, 1, 30, 10])
        # base_fluxes = ((np.arange(n_energy) + 1) * 1000**2)[::-1]
        fluxes = (
            base_fluxes[:, np.newaxis] * np.linspace(0.9, 1.1, n_pixels)[np.newaxis, :]
        )
        uncertainties = np.sqrt(fluxes)

        corrected_fluxes, corrected_unc, n_iter = corr.predictor_corrector_iteration(
            fluxes,
            uncertainties,
            energies,
            max_iterations=20,
            convergence_threshold=0.005,
        )

        # Check output shapes
        assert corrected_fluxes.shape == (n_energy, n_pixels)
        assert corrected_unc.shape == (n_energy, n_pixels)
        assert n_iter.shape == (n_pixels,)

        # All pixels should converge
        assert np.all(n_iter < 20)
        assert np.all(n_iter > 0)

        # Corrected fluxes should be finite and positive
        assert np.all(np.isfinite(corrected_fluxes))
        assert np.all(corrected_fluxes > 0)

    def test_apply_flux_correction_2d_spatial(self, hi_coeffs_file):
        """Test applying correction to data with 2D spatial dimensions (like Lo)."""
        flux_corr = PowerLawFluxCorrector(hi_coeffs_file)

        # Create xarray DataArrays with 2D spatial dimensions
        # Shape: (energy=9, spin=36, elevation=4) - simulating Lo's structure
        n_energy = 9
        n_spin = 36
        n_elev = 4

        energies_data = np.arange(n_energy) + 1
        flux_data = np.random.rand(n_energy, n_spin, n_elev) * 1000 + 100
        delta_flux_data = np.sqrt(flux_data)

        flux = xr.DataArray(
            flux_data,
            dims=["energy", "spin", "elevation"],
            coords={
                "energy": energies_data,
                "spin": np.arange(n_spin),
                "elevation": np.arange(n_elev),
            },
        )
        delta_flux = xr.DataArray(
            delta_flux_data,
            dims=["energy", "spin", "elevation"],
            coords={
                "energy": energies_data,
                "spin": np.arange(n_spin),
                "elevation": np.arange(n_elev),
            },
        )
        energies = xr.DataArray(
            energies_data, dims=["energy"], coords={"energy": energies_data}
        )

        # Apply correction
        corrected_flux, corrected_unc = flux_corr.apply_flux_correction(
            flux, delta_flux, energies
        )

        # Verify output has same dimensions and shape as input
        assert corrected_flux.dims == flux.dims
        assert corrected_unc.dims == delta_flux.dims
        assert corrected_flux.shape == flux.shape
        assert corrected_unc.shape == delta_flux.shape

        # Verify dimension order is preserved
        assert corrected_flux.dims == ("energy", "spin", "elevation")
        assert corrected_unc.dims == ("energy", "spin", "elevation")

        # Verify coordinates are preserved
        np.testing.assert_array_equal(corrected_flux.coords["energy"], energies_data)
        np.testing.assert_array_equal(corrected_flux.coords["spin"], np.arange(n_spin))
        np.testing.assert_array_equal(
            corrected_flux.coords["elevation"], np.arange(n_elev)
        )

        # Corrected values should be finite and mostly positive
        assert np.all(np.isfinite(corrected_flux))
        assert np.sum(corrected_flux > 0) > 0.9 * corrected_flux.size


@pytest.fixture
def hi_pset_cdf_path(imap_tests_path):
    """Path to test Hi PSET CDF file."""
    return imap_tests_path / "hi/data/l1/imap_hi_l1c_45sensor-pset_20250415_v999.cdf"


@pytest.fixture
def mock_hi_pset():
    """Create a minimal mock Hi pointing set dataset for testing."""
    # Create a simple dataset with necessary fields
    n_epoch = 1
    n_spin = 100

    data = xr.Dataset(
        {
            "epoch": (["epoch"], np.array([797949131184000000])),
            "epoch_delta": (["epoch"], np.array([1e12])),
            "hae_longitude": (
                ["epoch", "spin_angle_bin"],
                np.linspace(0, 360, n_spin, endpoint=False).reshape(n_epoch, n_spin),
            ),
            "hae_latitude": (
                ["epoch", "spin_angle_bin"],
                np.linspace(-90, 90, n_spin).reshape(n_epoch, n_spin),
            ),
            "spin_angle_bin": (["spin_angle_bin"], np.arange(n_spin)),
        },
        attrs={"Logical_source": "imap_hi_l1c_45sensor-pset"},
    )

    return data


@pytest.fixture
def mock_lo_pset():
    """Create a minimal mock Lo pointing set dataset for testing."""
    # Create a simple dataset with necessary fields for Lo
    n_epoch = 1
    n_spin = 50
    n_off = 20

    data = xr.Dataset(
        {
            "epoch": (["epoch"], np.array([797949131184000000])),
            "pointing_start_met": (["epoch"], np.array([100.0])),
            "pointing_end_met": (["epoch"], np.array([200.0])),
            "hae_longitude": (
                ["epoch", "spin_angle_bin", "off_angle_bin"],
                np.linspace(0, 360, n_spin * n_off, endpoint=False).reshape(
                    n_epoch, n_spin, n_off
                ),
            ),
            "hae_latitude": (
                ["epoch", "spin_angle_bin", "off_angle_bin"],
                np.linspace(-90, 90, n_spin * n_off).reshape(n_epoch, n_spin, n_off),
            ),
        },
        attrs={"Logical_source": "imap_lo_l1c_pset"},
    )

    return data


class TestComptonGettingCorrection:
    """Test suite for Compton-Getting correction functions."""

    @mock.patch("imap_processing.ena_maps.utils.corrections.ttj2000ns_to_et")
    @mock.patch("imap_processing.ena_maps.utils.corrections.geometry.imap_state")
    def test_add_spacecraft_position_and_velocity_to_pset(
        self, mock_imap_state, mock_ttj2000_to_et, mock_hi_pset
    ):
        """Test that spacecraft position and velocity are correctly added to pset."""
        # Mock conversion from TTJ2000ns to ET
        et = 1000.0
        mock_ttj2000_to_et.return_value = et
        # Mock spacecraft state vector (position + velocity in HAE frame)
        mock_sc_state = np.array([1e8, 2e8, 3e8, 10.0, 20.0, 30.0])  # km and km/s
        mock_imap_state.return_value = mock_sc_state

        mock_hi_pset = add_spacecraft_position_and_velocity_to_pset(mock_hi_pset)

        # Verify SPICE was called correctly
        mock_imap_state.assert_called_once_with(
            et, ref_frame=geometry.SpiceFrame.IMAP_HAE
        )

        # Verify sc_velocity was added
        assert "sc_velocity" in mock_hi_pset
        assert isinstance(mock_hi_pset["sc_velocity"], xr.DataArray)
        np.testing.assert_array_equal(
            mock_hi_pset["sc_velocity"].values, np.array([10.0, 20.0, 30.0])
        )

        # Verify sc_position was added
        assert "sc_position" in mock_hi_pset
        assert isinstance(mock_hi_pset["sc_position"], xr.DataArray)
        np.testing.assert_array_equal(
            mock_hi_pset["sc_position"].values, np.array([1e8, 2e8, 3e8])
        )

    @mock.patch("imap_processing.ena_maps.utils.corrections.ttj2000ns_to_et")
    @mock.patch("imap_processing.ena_maps.utils.corrections.geometry.imap_state")
    def test_add_spacecraft_position_and_velocity_to_pset_lo(
        self, mock_imap_state, mock_ttj2000_to_et, mock_lo_pset
    ):
        """Test that S/C position and velocity are correctly added to Lo pset."""
        # Mock conversion from TTJ2000ns to ET
        et = 1000.0
        mock_ttj2000_to_et.return_value = et
        # Mock spacecraft state vector (position + velocity in HAE frame)
        mock_sc_state = np.array([1e8, 2e8, 3e8, 15.0, 25.0, 35.0])  # km and km/s
        mock_imap_state.return_value = mock_sc_state

        # For Lo, pointing duration is calculated from MET times
        # pointing_end_met - pointing_start_met = 200.0 - 100.0 = 100.0 seconds
        # In nanoseconds: 100.0 * 1e9 = 1e11 ns
        # Midpoint: epoch + pointing_duration_ns / 2
        expected_midpoint_time_ns = mock_lo_pset["epoch"].values[0] + 1e11 / 2

        mock_lo_pset = add_spacecraft_position_and_velocity_to_pset(mock_lo_pset)

        # Verify SPICE was called correctly
        mock_ttj2000_to_et.assert_called_once_with(expected_midpoint_time_ns)
        mock_imap_state.assert_called_once_with(
            et, ref_frame=geometry.SpiceFrame.IMAP_HAE
        )

        # Verify sc_velocity was added
        assert "sc_velocity" in mock_lo_pset
        assert isinstance(mock_lo_pset["sc_velocity"], xr.DataArray)
        np.testing.assert_array_equal(
            mock_lo_pset["sc_velocity"].values, np.array([15.0, 25.0, 35.0])
        )

        # Verify sc_position was added
        assert "sc_position" in mock_lo_pset
        assert isinstance(mock_lo_pset["sc_position"], xr.DataArray)
        np.testing.assert_array_equal(
            mock_lo_pset["sc_position"].values, np.array([1e8, 2e8, 3e8])
        )

    def test_add_spacecraft_position_and_velocity_unsupported_instrument(self):
        """Test that unsupported instrument raises NotImplementedError."""
        # Create a dataset with unsupported Logical_source
        unsupported_pset = xr.Dataset(
            {
                "epoch": (["epoch"], np.array([797949131184000000])),
                "epoch_delta": (["epoch"], np.array([1e12])),
            },
            attrs={"Logical_source": "imap_unsupported_instrument_pset"},
        )

        with pytest.raises(NotImplementedError, match="does not support PSETs"):
            add_spacecraft_position_and_velocity_to_pset(unsupported_pset)

    def test_add_spacecraft_position_and_velocity_zero_duration(self, mock_hi_pset):
        """Test that zero pointing duration sets pos and velocity to zero vectors."""
        # Set epoch_delta to zero to simulate an empty/filtered pointing set
        mock_hi_pset["epoch_delta"] = xr.DataArray(np.array([0.0]), dims=["epoch"])

        result = add_spacecraft_position_and_velocity_to_pset(mock_hi_pset)

        # Both sc_velocity and sc_position should be zero vectors
        np.testing.assert_array_equal(result["sc_velocity"].values, np.zeros(3))
        np.testing.assert_array_equal(result["sc_position"].values, np.zeros(3))

    def test_add_cartesian_look_direction(self, mock_hi_pset):
        """Test that look directions are correctly calculated and added."""
        mock_hi_pset = _add_cartesian_look_direction(mock_hi_pset)

        # _add_cartesian_look_direction is just a wrapper around
        # geometry.spherical_to_cartesian. We only need to test that the
        # look_direction variable was added and has the correct shape.
        # Verify look_direction was added
        assert "look_direction" in mock_hi_pset
        assert isinstance(mock_hi_pset["look_direction"], xr.DataArray)

        # Verify shape
        expected_shape = (1, 100, 3)  # (epoch, spin_angle_bin, x_y_z)
        assert mock_hi_pset["look_direction"].shape == expected_shape

    @mock.patch("imap_processing.ena_maps.utils.corrections.geometry.imap_state")
    def test_calculate_compton_getting_transform(self, mock_imap_state, mock_hi_pset):
        """Test Compton-Getting transformation calculations."""
        # Set up spacecraft velocity
        mock_sc_state = np.array([1e8, 2e8, 3e8, 10.0, 20.0, 30.0])
        mock_imap_state.return_value = mock_sc_state

        mock_hi_pset = add_spacecraft_position_and_velocity_to_pset(mock_hi_pset)
        mock_hi_pset = _add_cartesian_look_direction(mock_hi_pset)

        # Create energy array
        energy_hf = xr.DataArray(
            np.array([500.0, 1000.0, 2000.0]),
            dims=["esa_energy_step"],
            coords={"esa_energy_step": [1, 2, 3]},
        )

        mock_hi_pset = _calculate_compton_getting_transform(mock_hi_pset, energy_hf)

        # Verify required variables were added
        assert "energy_hf" in mock_hi_pset
        assert "energy_sc" in mock_hi_pset
        assert "hae_longitude" in mock_hi_pset
        assert "hae_latitude" in mock_hi_pset

        # Verify energy_hf matches input
        np.testing.assert_array_equal(
            mock_hi_pset["energy_hf"].values, energy_hf.values
        )

        # Verify energy_sc has correct shape
        # The transformation should broadcast across energy and spatial dimensions
        assert "energy_sc" in mock_hi_pset
        energy_sc = mock_hi_pset["energy_sc"]
        # Shape should include energy dimension
        assert len(energy_sc.dims) >= 2

        # Verify energy_sc values are positive and reasonable
        assert np.all(energy_sc.values > 0)
        assert np.all(np.isfinite(energy_sc.values))

        # Verify corrected coordinates are within valid ranges
        assert np.all(mock_hi_pset["hae_longitude"].values >= 0)
        assert np.all(mock_hi_pset["hae_longitude"].values <= 360)
        assert np.all(mock_hi_pset["hae_latitude"].values >= -90)
        assert np.all(mock_hi_pset["hae_latitude"].values <= 90)

    @mock.patch("imap_processing.ena_maps.utils.corrections.geometry.imap_state")
    def test_apply_compton_getting_correction(self, mock_imap_state, mock_hi_pset):
        """Test full Compton-Getting correction pipeline."""
        # Set up spacecraft velocity
        mock_sc_state = np.array([1e8, 2e8, 3e8, 10.0, 20.0, 30.0])
        mock_imap_state.return_value = mock_sc_state

        # Create energy array
        energy_hf = xr.DataArray(
            np.array([500.0, 1000.0, 2000.0]),
            dims=["esa_energy_step"],
            coords={"esa_energy_step": [1, 2, 3]},
        )

        # add the required sc_velocity to the pointing set
        mock_hi_pset = add_spacecraft_position_and_velocity_to_pset(mock_hi_pset)

        # Apply the full correction
        mock_hi_pset = apply_compton_getting_correction(mock_hi_pset, energy_hf)

        # Verify all intermediate variables were added
        assert "sc_velocity" in mock_hi_pset
        assert "look_direction" in mock_hi_pset
        assert "energy_hf" in mock_hi_pset
        assert "energy_sc" in mock_hi_pset
        assert "hae_longitude" in mock_hi_pset
        assert "hae_latitude" in mock_hi_pset

    @pytest.mark.external_test_data
    def test_compton_getting_with_real_pset(self, hi_pset_cdf_path):
        """Test Compton-Getting correction with real Hi PSET data."""
        # Load real pointing set
        pset = load_cdf(hi_pset_cdf_path)
        pset = pset.rename(HiPointingSet.l1c_to_l2_var_mapping)

        # Store original coordinates for comparison
        original_lon = pset["hae_longitude"].copy()

        # Mock spacecraft state
        pset["sc_velocity"] = xr.DataArray(
            np.array([12.0, -27.0, 0.02]), dims=[CoordNames.CARTESIAN_VECTOR.value]
        )

        # Create energy array (Hi has 9 energy steps)
        energy_hf = xr.DataArray(
            np.array(
                [500.0, 750.0, 1100.0, 1650.0, 2500.0, 3750.0, 5700.0, 8520.0, 12800.0]
            ),
            dims=["esa_energy_step"],
            coords={"esa_energy_step": np.arange(1, 10)},
        )

        # Apply correction (pass the dataset, not the pointing set object)
        pset = apply_compton_getting_correction(pset, energy_hf)

        # Verify coordinates were modified
        corrected_lon = pset["hae_longitude"]
        corrected_lat = pset["hae_latitude"]

        # Shape should now include energy dimension
        assert "esa_energy_step" in corrected_lon.dims
        assert "esa_energy_step" in corrected_lat.dims
        assert corrected_lon.dims == (
            original_lon.dims[0],
            "esa_energy_step",
            original_lon.dims[1],
        )

        # Verify all values are in valid ranges
        assert np.all(corrected_lon.values >= 0)
        assert np.all(corrected_lon.values <= 360)
        assert np.all(corrected_lat.values >= -90)
        assert np.all(corrected_lat.values <= 90)

    def test_compton_getting_physical_consistency(self, mock_hi_pset):
        """Test physical consistency of Compton-Getting correction."""
        # Set up a known spacecraft velocity
        sc_velocity = np.array([30.0, 0.0, 0.0])  # Moving in +X direction at 30 km/s

        mock_hi_pset["sc_velocity"] = xr.DataArray(sc_velocity, dims=["x_y_z"])
        mock_hi_pset["sc_direction_vector"] = xr.DataArray(
            sc_velocity / np.linalg.norm(sc_velocity), dims=["x_y_z"]
        )

        # Set up simple look directions
        mock_hi_pset = _add_cartesian_look_direction(mock_hi_pset)

        # Single energy level
        energy_hf = xr.DataArray(np.array([1000.0]), dims=["esa_energy_step"])

        _calculate_compton_getting_transform(mock_hi_pset, energy_hf)

        # Physical checks:
        # 1. Energy in spacecraft frame should be higher for particles coming
        #    from the direction of spacecraft motion (ram direction)
        energy_sc = mock_hi_pset["energy_sc"]

        # 2. All energies should be positive
        assert np.all(energy_sc.values > 0)

        # 3. The spacecraft frame energy should differ from heliosphere frame
        # (they should not all be exactly 1000 eV)
        assert not np.allclose(energy_sc.values, 1000.0)

        # 4. Energy variation should exist across different look directions
        assert energy_sc.values.std() > 0


class TestRamMask:
    """Test suite for calculate_ram_mask function."""

    def test_ram_mask_calculation(self):
        """Test ram_mask correctly identifies ram and anti-ram directions."""
        # Create a simple mock pset with specific look directions
        n_directions = 4
        dataset = xr.Dataset(
            {
                "epoch": (["epoch"], np.array([797949131184000000])),
                # Set up specific look directions:
                # 0 degrees lon, 0 lat = +X direction (ram)
                # 180 degrees lon, 0 lat = -X direction (anti-ram)
                # 90 degrees lon, 0 lat = +Y direction (perpendicular)
                # 270 degrees lon, 0 lat = -Y direction (perpendicular)
                "hae_longitude": (
                    ["epoch", "direction"],
                    np.array([[0.0, 180.0, 90.0, 270.0]]),
                ),
                "hae_latitude": (
                    ["epoch", "direction"],
                    np.array([[0.0, 0.0, 0.0, 0.0]]),
                ),
                "direction": (["direction"], np.arange(n_directions)),
            }
        ).transpose("epoch", "direction")

        # Set up spacecraft velocity in +X direction
        sc_velocity = np.array([30.0, 0.0, 0.0])  # km/s
        dataset["sc_velocity"] = xr.DataArray(sc_velocity, dims=["x_y_z"])

        # Add look directions
        dataset = _add_cartesian_look_direction(dataset)

        # Single energy level
        energy_hf = xr.DataArray(np.array([1000.0]), dims=["esa_energy_step"])

        # Calculate CG transform
        dataset = _calculate_compton_getting_transform(dataset, energy_hf)

        dataset = calculate_ram_mask(dataset)

        # Verify ram_mask exists
        assert "ram_mask" in dataset
        ram_mask = dataset["ram_mask"]

        # Verify dimensions
        assert set(ram_mask.dims) == {"epoch", "esa_energy_step", "direction"}

        # Extract the mask values for easier checking
        mask_values = ram_mask.values.squeeze()

        # Direction 0 (0 deg lon, 0 lat = +X): Should be ram (True)
        # Direction 1 (180 deg lon, 0 lat = -X): Should be anti-ram (False)
        # Directions 2 and 3 (perpendicular): Will always be anti-ram (False)

        # The key test: particles coming from the spacecraft's direction of motion
        # (opposite to velocity) should be anti-ram (False)
        assert not mask_values[1], (
            "Particles from -X (opposite velocity) should be anti-ram"
        )

        # Particles coming from the direction the spacecraft is moving toward
        # should be ram (True)
        assert mask_values[0], "Particles from +X (along velocity) should be ram"

        # Particles coming from the perpendicular direction should always shift
        # to be coming from a slightly anti-ram direction
        assert not mask_values[2], "Particles from +Y should be anti-ram"
        assert not mask_values[3], "Particles from -Y should be anti-ram"

        # Verify all values are boolean
        assert ram_mask.dtype == bool

    @staticmethod
    def create_synthetic_pset_with_hae_coords(shape=(10, 20)):
        """Create a synthetic dataset with known HAE coordinates."""
        # Create longitude and latitude grids
        lons = np.linspace(0, 360, shape[0], endpoint=False)
        lats = np.linspace(-90, 90, shape[1])
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")

        # Create a minimal dataset
        dataset = xr.Dataset(
            {
                "hae_longitude": xr.DataArray(
                    lon_grid[np.newaxis, :, :],
                    dims=["epoch", "spin_angle", "off_angle"],
                ),
                "hae_latitude": xr.DataArray(
                    lat_grid[np.newaxis, :, :],
                    dims=["epoch", "spin_angle", "off_angle"],
                ),
            },
            coords={
                "epoch": xr.DataArray([1e18], dims=["epoch"]),
            },
        )

        return dataset

    def test_update_ram_mask_plus_x_direction(self):
        """Test RAM mask with spacecraft velocity in +X direction."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Spacecraft velocity in +X direction (HAE frame)
        pset["sc_velocity"] = np.array([1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)

        lon = pset["hae_longitude"].values
        ram_mask = pset["ram_mask"].values

        # For +X direction, RAM should be anywhere that the longitude is between
        # -90 and +90.
        idx_ram = np.nonzero((lon < 90) | (lon > 270))
        assert np.all(ram_mask[idx_ram]), (
            "Expected lon < 90 or lon > 270 to be RAM for +X velocity"
        )

        # Pixels with 90 < lon < 270 should be anti-RAM (pointing in -X direction)
        idx_anti = np.nonzero((lon > 90) & (lon < 270))
        assert not np.any(ram_mask[idx_anti]), (
            "Expected 90 < lon < 270 to be anti-RAM for +X velocity"
        )

    def test_update_ram_mask_minus_x_direction(self):
        """Test RAM mask with spacecraft velocity in -X direction."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Spacecraft velocity in -X direction (HAE frame)
        pset["sc_velocity"] = np.array([-1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)

        lon = pset["hae_longitude"].values
        ram_mask = pset["ram_mask"].values

        # For -X direction, anti-RAM should be anywhere that 90 < lon < 270.
        idx_ram = np.nonzero((lon > 90) & (lon < 270))
        assert np.all(ram_mask[idx_ram]), (
            "Expected 90 < lon < 270 to be RAM for -X velocity"
        )

        # Pixels with lon < 90 or lon > 270 should be RAM (pointing in -X direction)
        idx_anti = np.nonzero((lon < 90) | (lon > 270))
        assert not np.any(ram_mask[idx_anti]), (
            "Expected lon < 90 or lon > 270 to be anit-RAM for -X velocity"
        )

    def test_update_ram_mask_plus_y_direction(self):
        """Test RAM mask with spacecraft velocity in +Y direction."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Spacecraft velocity in +Y direction (HAE frame)
        pset["sc_velocity"] = np.array([0.0, 1.0, 0.0])
        pset = calculate_ram_mask(pset)

        lon = pset["hae_longitude"].values
        ram_mask = pset["ram_mask"].values

        # For +Y direction, RAM should be anywhere that the 0 < lon < 180.
        idx_ram = np.nonzero((0 < lon) & (lon < 180))
        assert np.all(ram_mask[idx_ram]), "Expected lat > 0 to be RAM for +Y velocity"

        # Pixels with lon > 180 should be anti-RAM (pointing in -Y direction)
        idx_anti = np.nonzero(lon > 180)
        assert not np.any(ram_mask[idx_anti]), (
            "Expected lat < 0 to be anti-RAM for +Y velocity"
        )

    def test_update_ram_mask_magnitude_invariance(self):
        """Test that RAM mask is invariant to velocity vector magnitude."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Test with two different magnitudes in the same direction
        pset["sc_velocity"] = np.array([1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)
        ram_mask_1 = pset["ram_mask"].values.copy()

        pset["sc_velocity"] = np.array([100.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)
        ram_mask_2 = pset["ram_mask"].values.copy()

        # The masks should be identical since direction is the same
        np.testing.assert_array_equal(ram_mask_1, ram_mask_2)

    def test_update_ram_mask_dot_product_correctness(self):
        """Test that dot product calculation is mathematically correct."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Use a simple spacecraft velocity vector
        pset["sc_velocity"] = np.array([1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)

        # Manually verify a few specific pixels
        lon = pset["hae_longitude"].values
        lat = pset["hae_latitude"].values
        ram_mask = pset["ram_mask"].values

        # Test pixel at lon=0, lat=0 (should point in +X direction)
        lon_rad = np.deg2rad(0)
        lat_rad = np.deg2rad(0)
        x = np.cos(lat_rad) * np.cos(lon_rad)  # = 1.0
        dot_product = x * 1.0  # = 1.0
        expected_ram = dot_product >= 0  # True

        idx = np.where((np.abs(lon - 0) < 1) & (np.abs(lat - 0) < 1))
        if idx[0].size > 0:
            assert ram_mask[idx][0] == expected_ram

    def test_update_ram_mask_dimensions_preserved(self):
        """Test that update_ram_mask preserves coordinate dimensions."""
        # Test with 2D spatial dimensions (like LoPointingSet)
        pset_2d = self.create_synthetic_pset_with_hae_coords(shape=(5, 10))

        # Get original dimensions
        original_dims_2d = pset_2d["hae_longitude"].dims

        # Update ram_mask
        pset_2d["sc_velocity"] = np.array([1.0, 1.0, 0.0])
        pset_2d = calculate_ram_mask(pset_2d)

        # Verify dimensions are preserved
        assert pset_2d["ram_mask"].dims == original_dims_2d

        # Test with 1D spatial dimensions (like HiPointingSet)
        # Create synthetic dataset with 1D spatial dimension
        lons = np.linspace(0, 360, 20, endpoint=False)
        lats = np.linspace(-90, 90, 20)

        dataset_1d = xr.Dataset(
            {
                "hae_longitude": xr.DataArray(
                    lons[np.newaxis, :],
                    dims=["epoch", "spin_angle_bin"],
                ),
                "hae_latitude": xr.DataArray(
                    lats[np.newaxis, :],
                    dims=["epoch", "spin_angle_bin"],
                ),
            },
            coords={
                "epoch": xr.DataArray([1e18], dims=["epoch"]),
            },
        )

        # Get original dimensions
        original_dims_1d = dataset_1d["hae_longitude"].dims

        # Update ram_mask
        dataset_1d["sc_velocity"] = np.array([1.0, 1.0, 0.0])
        dataset_1d = calculate_ram_mask(dataset_1d)

        # Verify dimensions are preserved
        assert dataset_1d["ram_mask"].dims == original_dims_1d

    def test_update_ram_mask_replaces_existing(self):
        """Test that update_ram_mask replaces existing ram_mask."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Set initial mask with +X direction
        pset["sc_velocity"] = np.array([1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)
        ram_mask_1 = pset["ram_mask"].values.copy()

        # Update mask with opposite direction
        pset["sc_velocity"] = np.array([-1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)
        ram_mask_2 = pset["ram_mask"].values.copy()

        # The masks should be different
        assert not np.array_equal(ram_mask_1, ram_mask_2)

    def test_update_ram_mask_arbitrary_direction(self):
        """Test RAM mask with arbitrary spacecraft velocity direction."""
        pset = self.create_synthetic_pset_with_hae_coords(shape=(36, 18))

        # Use an arbitrary direction (not aligned with axes)
        pset["sc_velocity"] = np.array([1.0, 1.0, 0.5])
        pset = calculate_ram_mask(pset)

        # Verify the mask was created
        assert "ram_mask" in pset

        # Verify approximately half the pixels are RAM (for a sphere)
        ram_fraction = pset["ram_mask"].sum().values / pset["ram_mask"].size
        # Should be close to 0.5, allowing for discretization effects
        np.testing.assert_allclose(ram_fraction, 0.5, atol=0.05)


class TestInterpolateMapFluxToHelioFrame:
    """Test suite for interpolate_map_flux_to_helio_frame function."""

    def create_test_map_dataset(self, n_energy=5, n_spatial=10, power_law_slope=-2.0):
        """Create a synthetic map dataset for testing interpolation.

        Parameters
        ----------
        n_energy : int
            Number of energy channels
        n_spatial : int
            Number of spatial pixels
        power_law_slope : float
            Power-law spectral index for test flux

        Returns
        -------
        tuple
            (map_ds, esa_energies, helio_energies)
        """
        # Define ESA energy channels (in eV)
        esa_energies = np.array([500.0, 1000.0, 2000.0, 4000.0, 8000.0])[:n_energy]

        # Create flux with a simple power-law spectrum: flux = E^slope
        flux_base = esa_energies[:, np.newaxis] ** power_law_slope

        # Add some spatial variation (multiply by factors between 0.5 and 1.5)
        spatial_factors = np.linspace(0.5, 1.5, n_spatial)
        flux = flux_base * spatial_factors

        # Create uncertainties (10% statistical, 5% systematic)
        stat_unc = 0.1 * flux
        sys_err = 0.05 * flux

        # Create spacecraft energies slightly different from ESA energies
        # to simulate Compton-Getting shift
        # Add a small spatial-dependent shift
        energy_shift_factor = 1.0 + 0.1 * np.linspace(-1, 1, n_spatial)
        energy_sc = esa_energies[:, np.newaxis] * energy_shift_factor

        # Create xarray Dataset
        map_ds = xr.Dataset(
            {
                "ena_intensity": (["energy", "spatial"], flux),
                "ena_intensity_stat_uncert": (["energy", "spatial"], stat_unc),
                "ena_intensity_sys_err": (["energy", "spatial"], sys_err),
                "energy_sc": (["energy", "spatial"], energy_sc),
            },
            coords={
                "energy": np.arange(n_energy),
                "spatial": np.arange(n_spatial),
            },
        )

        # Helio energies are the same as ESA energies (standard case)
        helio_energies = xr.DataArray(
            esa_energies,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )

        esa_energies_da = xr.DataArray(
            esa_energies,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )

        return map_ds, esa_energies_da, helio_energies

    def test_basic_interpolation(self):
        """Test basic functionality of interpolation."""

        map_ds, esa_energies, helio_energies = self.create_test_map_dataset()

        # Apply interpolation
        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # Verify output structure
        assert "ena_intensity" in result_ds
        assert "ena_intensity_stat_uncert" in result_ds
        assert "ena_intensity_sys_err" in result_ds

        # Verify shapes are preserved
        assert result_ds["ena_intensity"].shape == map_ds["ena_intensity"].shape
        assert (
            result_ds["ena_intensity_stat_uncert"].shape
            == map_ds["ena_intensity_stat_uncert"].shape
        )
        assert (
            result_ds["ena_intensity_sys_err"].shape
            == map_ds["ena_intensity_sys_err"].shape
        )

    def test_energy_unit_insensitivity(self):
        """Test that units of eV or keV produce the same result."""

        map_ds, esa_energies, helio_energies = self.create_test_map_dataset()

        # Apply interpolation
        ev_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )
        kev_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies / 1000, helio_energies / 1000, ["ena_intensity"]
        )

        # Verify results are the same
        xr.testing.assert_equal(ev_ds, kev_ds)

    def test_power_law_interpolation_accuracy(self):
        """Test that power-law interpolation formula is correct."""

        # Create simple test case with known power-law
        # flux = E^(-2)
        power_law_slope = -2.0
        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=3, n_spatial=1, power_law_slope=power_law_slope
        )

        # Manually set energy_sc to be between two ESA energies
        # ESA energies: [500, 1000, 2000]
        # Set energy_sc for middle channel to 750 eV (between 500 and 1000)
        map_ds["energy_sc"].values[1, 0] = 750.0

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # For a perfect power-law with flux = E^(-2) * spatial_factor:
        # With n_spatial=1, spatial_factor = 0.5 (from np.linspace(0.5, 1.5, 1))
        # The interpolation process does:
        # 1. Interpolates flux at E_sc=750 from values at 500 and 1000
        #    Expected: 750^(-2) * 0.5
        # 2. Scales to E_helio=1000: flux * (1000/750)
        #    = 750^(-2) * 0.5 * (1000/750)

        # Calculate expected result for middle energy channel
        e_sc = 750.0
        e_helio = 1000.0
        spatial_factor = 0.5  # From create_test_map_dataset with n_spatial=1
        expected_flux_middle = (
            (e_sc**power_law_slope) * (e_helio / e_sc) * spatial_factor
        )
        unc_factor = np.log(e_sc / 500) / np.log(1000 / 500)
        expected_stat_uncert = expected_flux_middle * np.sqrt(
            0.1**2 * (1 + unc_factor**2) + unc_factor**2 * 0.1**2
        )

        # Compare interpolated result to expected value
        # (should be very close for a perfect power-law)
        np.testing.assert_allclose(
            result_ds["ena_intensity"].values[1, 0],
            expected_flux_middle,
            rtol=1e-10,
        )

        # Check expected stat. unc.
        np.testing.assert_allclose(
            result_ds["ena_intensity_stat_uncert"].values[1, 0],
            expected_stat_uncert,
            rtol=1e-10,
        )

        # The flux should be finite and positive
        assert np.all(np.isfinite(result_ds["ena_intensity"].values))
        assert np.all(result_ds["ena_intensity"].values > 0)

    def test_statistical_uncertainty_propagation(self):
        """Test that statistical uncertainty follows Equation 75."""

        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=3, n_spatial=1
        )

        # Set up a specific case where we can verify the formula
        # Set energy_sc to midpoint between two channels
        e_sc = 1400.0  # Between left and right

        map_ds["energy_sc"].values[1, 0] = e_sc

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # Statistical uncertainty should be positive and finite
        stat_unc = result_ds["ena_intensity_stat_uncert"].values
        assert np.all(stat_unc >= 0)
        assert np.all(np.isfinite(stat_unc))

        # Statistical uncertainty should scale with flux
        # (relative uncertainty should be similar to input)
        flux = result_ds["ena_intensity"].values
        rel_unc_output = stat_unc / flux

        # Should be on the order of input relative uncertainty (10%)
        # Allow for propagation effects
        assert np.all(rel_unc_output > 0)
        assert np.all(rel_unc_output < 1.0)  # Should be reasonable

    def test_systematic_uncertainty_propagation(self):
        """Test that systematic uncertainty follows Equation 76."""

        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=3, n_spatial=1
        )

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # Systematic uncertainty should be positive and finite
        sys_err = result_ds["ena_intensity_sys_err"].values
        assert np.all(sys_err >= 0)
        assert np.all(np.isfinite(sys_err))

        # Systematic error should scale proportionally with flux
        flux = result_ds["ena_intensity"].values
        rel_sys_err = sys_err / flux

        # Relative systematic error should be preserved (5% in input)
        # within reasonable tolerance for the transformations
        assert np.all(rel_sys_err > 0)
        assert np.all(rel_sys_err < 0.5)  # Should be reasonable

    def test_systematic_uncertainty_update_flag(self):
        """Test that systematic error is unchanged when flag is set False."""

        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=3, n_spatial=1
        )
        sys_err_input = map_ds["ena_intensity_sys_err"].copy()

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds,
            esa_energies,
            helio_energies,
            ["ena_intensity"],
            update_sys_err=False,
        )

        # Systematic uncertainty should be positive and finite
        xr.testing.assert_equal(result_ds["ena_intensity_sys_err"], sys_err_input)

    def test_energy_scaling_transformation(self):
        """Test Liouville theorem: flux_helio = flux_sc * (E_helio / E_sc)."""

        # Create dataset where energy_sc equals ESA energies (no CG shift)
        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=3, n_spatial=1
        )

        # Set energy_sc exactly equal to ESA energies
        map_ds["energy_sc"].values[:, 0] = esa_energies.values

        # Store original flux for comparison
        original_flux = map_ds["ena_intensity"].values.copy()

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # When E_sc = E_esa and E_helio = E_esa, then E_helio/E_sc = 1
        # So flux should be approximately preserved (modulo interpolation effects)
        result_flux = result_ds["ena_intensity"].values

        # The ratio should be close to 1 for each pixel
        # (within numerical precision and interpolation effects)
        ratio = result_flux / original_flux

        # Allow for some numerical error and interpolation effects
        assert np.all(np.isfinite(ratio))
        # Most values should be reasonably close to original
        # (exact match not expected due to interpolation)
        np.testing.assert_allclose(ratio, 1)

    def test_infinite_values_converted_to_nan(self):
        """Test that infinite values are converted to NaN."""

        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=3, n_spatial=2
        )

        # Introduce a zero flux to create divide-by-zero
        map_ds["ena_intensity"].values[1, 0] = 0.0

        # Set energy_sc to zero to create potential infinities
        map_ds["energy_sc"].values[1, 1] = 0.0

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # Check that we have NaN values where expected, not infinities
        flux = result_ds["ena_intensity"].values
        stat_unc = result_ds["ena_intensity_stat_uncert"].values
        sys_err = result_ds["ena_intensity_sys_err"].values

        # Should have no infinities
        assert not np.any(np.isinf(flux))
        assert not np.any(np.isinf(stat_unc))
        assert not np.any(np.isinf(sys_err))

    def test_nan_input_propagation(self):
        """Test that NaN inputs properly propagate to NaN outputs."""
        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=3, n_spatial=3
        )

        # Set some input values to NaN
        map_ds["ena_intensity"].values[1, 0] = np.nan
        map_ds["ena_intensity_stat_uncert"].values[1, 1] = np.nan
        map_ds["ena_intensity_sys_err"].values[1, 2] = np.nan

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # NaN in flux should propagate to flux, stat_uncert, and sys_err
        # at that location
        assert np.isnan(result_ds["ena_intensity"].values[1, 0])
        assert np.isnan(result_ds["ena_intensity_stat_uncert"].values[1, 0])
        assert np.isnan(result_ds["ena_intensity_sys_err"].values[1, 0])

        # NaN in stat_uncert input should result in NaN stat_uncert output
        assert np.isnan(result_ds["ena_intensity_stat_uncert"].values[1, 1])

        # NaN in sys_err input should result in NaN sys_err output
        assert np.isnan(result_ds["ena_intensity_sys_err"].values[1, 2])

    def test_multidimensional_spatial_coords(self):
        """Test that interpolation works with multi-dimensional spatial coordinates."""

        n_energy = 4
        n_lat = 6
        n_lon = 8

        # Define ESA energies
        esa_energies_vals = np.array([500.0, 1000.0, 2000.0, 4000.0])

        # Create flux with spatial dimensions (lat, lon)
        power_law_slope = -2.0
        flux = np.zeros((n_energy, n_lat, n_lon))
        stat_unc = np.zeros((n_energy, n_lat, n_lon))
        sys_err = np.zeros((n_energy, n_lat, n_lon))
        energy_sc = np.zeros((n_energy, n_lat, n_lon))

        for i in range(n_energy):
            # Power-law flux with spatial variation
            spatial_pattern = 1.0 + 0.5 * np.random.random((n_lat, n_lon))
            flux[i, :, :] = (esa_energies_vals[i] ** power_law_slope) * spatial_pattern
            stat_unc[i, :, :] = 0.1 * flux[i, :, :]
            sys_err[i, :, :] = 0.05 * flux[i, :, :]

            # Energy shift varies with position
            energy_sc[i, :, :] = esa_energies_vals[i] * (
                1.0 + 0.1 * np.random.random((n_lat, n_lon))
            )

        # Create dataset with 2D spatial coordinates
        map_ds = xr.Dataset(
            {
                "ena_intensity": (["energy", "latitude", "longitude"], flux),
                "ena_intensity_stat_uncert": (
                    ["energy", "latitude", "longitude"],
                    stat_unc,
                ),
                "ena_intensity_sys_err": (["energy", "latitude", "longitude"], sys_err),
                "energy_sc": (["energy", "latitude", "longitude"], energy_sc),
            },
            coords={
                "energy": np.arange(n_energy),
                "latitude": np.arange(n_lat),
                "longitude": np.arange(n_lon),
            },
        )

        esa_energies = xr.DataArray(
            esa_energies_vals,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )
        helio_energies = esa_energies.copy()

        # Apply interpolation
        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # Verify output shape matches input
        assert result_ds["ena_intensity"].shape == (n_energy, n_lat, n_lon)
        assert result_ds["ena_intensity_stat_uncert"].shape == (n_energy, n_lat, n_lon)
        assert result_ds["ena_intensity_sys_err"].shape == (n_energy, n_lat, n_lon)

        # Verify dimensions are preserved
        assert list(result_ds["ena_intensity"].dims) == [
            "energy",
            "latitude",
            "longitude",
        ]

        # Verify values are reasonable
        assert np.all(result_ds["ena_intensity"].values > 0)
        assert np.all(np.isfinite(result_ds["ena_intensity"].values))

    def test_boundary_energy_channels(self):
        """Test interpolation behavior at energy boundaries."""

        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=5, n_spatial=3
        )

        # Test when energy_sc is below the lowest ESA energy
        map_ds["energy_sc"].values[0, 0] = 0.9 * esa_energies.values[0]

        # Test when energy_sc is above the highest ESA energy
        map_ds["energy_sc"].values[-1, -1] = 1.1 * esa_energies.values[-1]

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # Should handle boundary cases without errors
        flux = result_ds["ena_intensity"].values

        # Boundary pixels should have values (possibly NaN, but no crashes)
        # The function clips indices to valid range, so these should interpolate
        # using the boundary channels
        assert flux.shape == map_ds["ena_intensity"].shape

    def test_preserves_dataset_structure(self):
        """Test that the function preserves the dataset structure."""

        map_ds, esa_energies, helio_energies = self.create_test_map_dataset()

        # Store original attributes
        original_dims = list(map_ds["ena_intensity"].dims)
        original_coords = list(map_ds.coords.keys())

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # Verify dimensions are preserved
        assert list(result_ds["ena_intensity"].dims) == original_dims

        # Verify coordinates are preserved
        assert list(result_ds.coords.keys()) == original_coords

        # Verify it's still an xarray Dataset
        assert isinstance(result_ds, xr.Dataset)

    def test_with_uniform_flux(self):
        """Test interpolation with uniform flux (no spatial variation)."""

        n_energy = 4
        n_spatial = 5

        esa_energies_vals = np.array([500.0, 1000.0, 2000.0, 4000.0])

        # Create uniform flux (same for all spatial pixels)
        power_law_slope = -2.0
        flux_uniform = (esa_energies_vals**power_law_slope)[:, np.newaxis]
        flux = np.tile(flux_uniform, (1, n_spatial))

        stat_unc = 0.1 * flux
        sys_err = 0.05 * flux

        # Energy shift uniform across space
        energy_sc = np.tile(esa_energies_vals[:, np.newaxis] * 1.05, (1, n_spatial))

        map_ds = xr.Dataset(
            {
                "ena_intensity": (["energy", "spatial"], flux),
                "ena_intensity_stat_uncert": (["energy", "spatial"], stat_unc),
                "ena_intensity_sys_err": (["energy", "spatial"], sys_err),
                "energy_sc": (["energy", "spatial"], energy_sc),
            },
            coords={
                "energy": np.arange(n_energy),
                "spatial": np.arange(n_spatial),
            },
        )

        esa_energies = xr.DataArray(
            esa_energies_vals,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )
        helio_energies = esa_energies.copy()

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # With uniform input, output should also be uniform across spatial dimension
        result_flux = result_ds["ena_intensity"].values

        # Check that each energy channel has uniform values across spatial pixels
        for i_energy in range(n_energy):
            spatial_values = result_flux[i_energy, :]
            # All spatial pixels at this energy should be similar
            if np.all(np.isfinite(spatial_values)):
                std_dev = np.std(spatial_values)
                mean_val = np.mean(spatial_values)
                # Relative std should be very small for uniform input
                if mean_val > 0:
                    rel_std = std_dev / mean_val
                    assert rel_std < 1e-10, f"Energy {i_energy}: rel_std = {rel_std}"

    def test_multiple_variables_interpolation(self):
        """Test interpolation with multiple intensity variables."""
        # Create base dataset with ena_intensity
        map_ds, esa_energies, helio_energies = self.create_test_map_dataset(
            n_energy=3, n_spatial=5
        )

        # Add a second intensity variable (e.g., background) with different values
        map_ds["background_intensity"] = map_ds["ena_intensity"] * 0.2  # 20% of signal
        map_ds["background_intensity_stat_uncert"] = (
            map_ds["ena_intensity_stat_uncert"] * 0.2
        )
        map_ds["background_intensity_sys_err"] = map_ds["ena_intensity_sys_err"] * 0.2

        # Interpolate both variables
        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds,
            esa_energies,
            helio_energies,
            ["ena_intensity", "background_intensity"],
        )

        # Verify shapes are preserved for ena_intensity
        assert result_ds["ena_intensity"].shape == map_ds["ena_intensity"].shape
        assert (
            result_ds["ena_intensity_stat_uncert"].shape
            == map_ds["ena_intensity_stat_uncert"].shape
        )
        assert (
            result_ds["ena_intensity_sys_err"].shape
            == map_ds["ena_intensity_sys_err"].shape
        )

        # Verify shapes are preserved for background
        assert (
            result_ds["background_intensity"].shape
            == map_ds["background_intensity"].shape
        )
        assert (
            result_ds["background_intensity_stat_uncert"].shape
            == map_ds["background_intensity_stat_uncert"].shape
        )
        assert (
            result_ds["background_intensity_sys_err"].shape
            == map_ds["background_intensity_sys_err"].shape
        )

        # Verify all values are finite and positive
        assert np.all(np.isfinite(result_ds["ena_intensity"].values))
        assert np.all(result_ds["ena_intensity"].values > 0)
        assert np.all(np.isfinite(result_ds["background_intensity"].values))
        assert np.all(result_ds["background_intensity"].values > 0)

        # Verify the relative scaling between signal and background is
        # approximately preserved (background should still be ~20% of signal
        # after interpolation)
        signal_values = result_ds["ena_intensity"].values
        background_values = result_ds["background_intensity"].values
        ratio = background_values / signal_values
        # Allow for some numerical variation due to interpolation
        np.testing.assert_allclose(ratio, 0.2, rtol=0.01)

    def test_linear_fallback_when_flux_left_zero(self):
        """Test that linear interpolation is used when flux_left is zero."""
        n_energy = 3
        n_spatial = 2

        esa_energies_vals = np.array([500.0, 1000.0, 2000.0])

        # Create flux where one pixel has zero flux at energy index 0
        flux = np.array(
            [
                [0.0, 1.0],  # energy 0: pixel 0 is zero
                [1.0, 2.0],  # energy 1
                [0.5, 1.5],  # energy 2
            ]
        )
        stat_unc = 0.1 * np.maximum(flux, 0.1)
        sys_err = 0.05 * np.maximum(flux, 0.1)

        # Set energy_sc to be between energy channels 0 and 1
        energy_sc = np.array(
            [
                [750.0, 750.0],  # interpolating between 500 and 1000
                [1500.0, 1500.0],  # interpolating between 1000 and 2000
                [1800.0, 1800.0],  # near the boundary
            ]
        )

        map_ds = xr.Dataset(
            {
                "ena_intensity": (["energy", "spatial"], flux),
                "ena_intensity_stat_uncert": (["energy", "spatial"], stat_unc),
                "ena_intensity_sys_err": (["energy", "spatial"], sys_err),
                "energy_sc": (["energy", "spatial"], energy_sc),
            },
            coords={
                "energy": np.arange(n_energy),
                "spatial": np.arange(n_spatial),
            },
        )

        esa_energies = xr.DataArray(
            esa_energies_vals,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )
        helio_energies = esa_energies.copy()

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        # Verify results are finite (not NaN) where we expect valid values
        result_flux = result_ds["ena_intensity"].values

        # With linear fallback, pixel 0 at energy 0 should produce a valid result
        # because we're interpolating between 0 (at 500eV) and 1 (at 1000eV)
        # Linear interpolation at 750eV: 0 + (1-0) * (750-500)/(1000-500) = 0.5
        # Then energy scaling: 0.5 * (500/750) = 1/3
        expected_flux = 0.5 * (500.0 / 750.0)  # = 1/3
        np.testing.assert_allclose(
            result_flux[0, 0],
            expected_flux,
            rtol=1e-10,
            err_msg="Linear fallback should produce correct interpolated value",
        )

        # Statistical uncertainty should also be finite
        result_stat_unc = result_ds["ena_intensity_stat_uncert"].values
        assert np.isfinite(result_stat_unc[0, 0]), (
            "Statistical uncertainty should be finite with linear fallback"
        )

    def test_linear_fallback_when_flux_right_zero(self):
        """Test that linear interpolation is used when flux_right is zero."""
        n_energy = 3
        n_spatial = 2

        esa_energies_vals = np.array([500.0, 1000.0, 2000.0])

        # Create flux where one pixel has zero flux at energy index 1
        flux = np.array(
            [
                [1.0, 1.0],  # energy 0
                [0.0, 2.0],  # energy 1: pixel 0 is zero
                [0.5, 1.5],  # energy 2
            ]
        )
        stat_unc = 0.1 * np.maximum(flux, 0.1)
        sys_err = 0.05 * np.maximum(flux, 0.1)

        # Set energy_sc to be between energy channels 0 and 1
        energy_sc = np.array(
            [
                [750.0, 750.0],  # interpolating between 500 and 1000
                [1500.0, 1500.0],  # interpolating between 1000 and 2000
                [1800.0, 1800.0],  # near the boundary
            ]
        )

        map_ds = xr.Dataset(
            {
                "ena_intensity": (["energy", "spatial"], flux),
                "ena_intensity_stat_uncert": (["energy", "spatial"], stat_unc),
                "ena_intensity_sys_err": (["energy", "spatial"], sys_err),
                "energy_sc": (["energy", "spatial"], energy_sc),
            },
            coords={
                "energy": np.arange(n_energy),
                "spatial": np.arange(n_spatial),
            },
        )

        esa_energies = xr.DataArray(
            esa_energies_vals,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )
        helio_energies = esa_energies.copy()

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        result_flux = result_ds["ena_intensity"].values

        # With linear fallback, pixel 0 at energy 0 should produce a valid result
        # Linear interpolation from flux=1 at 500eV to flux=0 at 1000eV
        # At 750eV: 1 + (0-1) * (750-500)/(1000-500) = 1 - 0.5 = 0.5
        # Then energy scaling: 0.5 * (500/750) = 1/3
        expected_flux = 0.5 * (500.0 / 750.0)  # = 1/3
        np.testing.assert_allclose(
            result_flux[0, 0],
            expected_flux,
            rtol=1e-10,
            err_msg="Linear fallback should produce correct interpolated value",
        )

    def test_linear_fallback_when_both_fluxes_zero(self):
        """Test that both bounding fluxes being zero produces zero output."""
        n_energy = 3
        n_spatial = 2

        esa_energies_vals = np.array([500.0, 1000.0, 2000.0])

        # Create flux where pixel 0 has zero flux at both energy 0 and 1
        flux = np.array(
            [
                [0.0, 1.0],  # energy 0: pixel 0 is zero
                [0.0, 2.0],  # energy 1: pixel 0 is zero
                [0.5, 1.5],  # energy 2
            ]
        )
        stat_unc = 0.1 * np.maximum(flux, 0.1)
        sys_err = 0.05 * np.maximum(flux, 0.1)

        # Set energy_sc to be between energy channels 0 and 1 for first energy
        energy_sc = np.array(
            [
                [750.0, 750.0],  # interpolating between 500 and 1000
                [1500.0, 1500.0],
                [1800.0, 1800.0],
            ]
        )

        map_ds = xr.Dataset(
            {
                "ena_intensity": (["energy", "spatial"], flux),
                "ena_intensity_stat_uncert": (["energy", "spatial"], stat_unc),
                "ena_intensity_sys_err": (["energy", "spatial"], sys_err),
                "energy_sc": (["energy", "spatial"], energy_sc),
            },
            coords={
                "energy": np.arange(n_energy),
                "spatial": np.arange(n_spatial),
            },
        )

        esa_energies = xr.DataArray(
            esa_energies_vals,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )
        helio_energies = esa_energies.copy()

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        result_flux = result_ds["ena_intensity"].values

        # When both bounding fluxes are zero, linear interpolation gives 0
        # Result should be 0 (not NaN)
        assert result_flux[0, 0] == 0.0, (
            "When both bounding fluxes are zero, result should be 0"
        )

    def test_negative_interpolated_flux_clamped_to_zero(self):
        """Test that negative interpolated flux is clamped to zero."""
        n_energy = 3
        n_spatial = 1

        esa_energies_vals = np.array([500.0, 1000.0, 2000.0])

        # Create flux that decreases steeply - linear extrapolation could go negative
        flux = np.array(
            [
                [2.0],  # energy 0
                [0.1],  # energy 1: much smaller
                [0.5],  # energy 2
            ]
        )
        stat_unc = 0.1 * flux
        sys_err = 0.05 * flux

        # Set energy_sc beyond the range to trigger extrapolation
        # At energy index 0, set energy_sc below 500eV to extrapolate
        energy_sc = np.array(
            [
                [400.0],  # Below lowest ESA energy - will use channels 0,1
                [800.0],
                [1500.0],
            ]
        )

        map_ds = xr.Dataset(
            {
                "ena_intensity": (["energy", "spatial"], flux),
                "ena_intensity_stat_uncert": (["energy", "spatial"], stat_unc),
                "ena_intensity_sys_err": (["energy", "spatial"], sys_err),
                "energy_sc": (["energy", "spatial"], energy_sc),
            },
            coords={
                "energy": np.arange(n_energy),
                "spatial": np.arange(n_spatial),
            },
        )

        esa_energies = xr.DataArray(
            esa_energies_vals,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )
        helio_energies = esa_energies.copy()

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        result_flux = result_ds["ena_intensity"].values

        # All flux values should be non-negative
        assert np.all(result_flux >= 0), "All interpolated flux values should be >= 0"

    def test_linear_and_powerlaw_mixed(self):
        """Test correct interpolation method for mixed zero/positive flux."""
        n_energy = 3
        n_spatial = 4

        esa_energies_vals = np.array([500.0, 1000.0, 2000.0])

        # Create mixed flux data:
        # - Pixel 0: zero at left boundary (needs linear)
        # - Pixel 1: zero at right boundary (needs linear)
        # - Pixel 2: all positive (can use power-law)
        # - Pixel 3: zero at both (needs linear, result=0)
        flux = np.array(
            [
                [0.0, 1.0, 1.0, 0.0],  # energy 0
                [1.0, 0.0, 2.0, 0.0],  # energy 1
                [0.5, 0.5, 1.5, 0.5],  # energy 2
            ]
        )
        stat_unc = 0.1 * np.maximum(flux, 0.1)
        sys_err = 0.05 * np.maximum(flux, 0.1)

        # Set energy_sc to be between energy channels 0 and 1
        energy_sc = np.full((n_energy, n_spatial), 750.0)
        energy_sc[1, :] = 1500.0
        energy_sc[2, :] = 1800.0

        map_ds = xr.Dataset(
            {
                "ena_intensity": (["energy", "spatial"], flux),
                "ena_intensity_stat_uncert": (["energy", "spatial"], stat_unc),
                "ena_intensity_sys_err": (["energy", "spatial"], sys_err),
                "energy_sc": (["energy", "spatial"], energy_sc),
            },
            coords={
                "energy": np.arange(n_energy),
                "spatial": np.arange(n_spatial),
            },
        )

        esa_energies = xr.DataArray(
            esa_energies_vals,
            dims=["energy"],
            coords={"energy": np.arange(n_energy)},
        )
        helio_energies = esa_energies.copy()

        result_ds = interpolate_map_flux_to_helio_frame(
            map_ds, esa_energies, helio_energies, ["ena_intensity"]
        )

        result_flux = result_ds["ena_intensity"].values

        # Check that all pixels at energy 0 have finite, non-negative results
        assert np.all(np.isfinite(result_flux[0, :])), (
            "All pixels should have finite results"
        )
        assert np.all(result_flux[0, :] >= 0), "All flux values should be non-negative"

        # Pixel 0: Linear interpolation from 0 to 1 at 750eV should give positive
        assert result_flux[0, 0] > 0, (
            "Pixel 0 should have positive flux (linear from 0 to 1)"
        )

        # Pixel 1: Linear interpolation from 1 to 0 at 750eV should give positive
        assert result_flux[0, 1] > 0, (
            "Pixel 1 should have positive flux (linear from 1 to 0)"
        )

        # Pixel 2: Power-law interpolation should give positive
        assert result_flux[0, 2] > 0, "Pixel 2 should have positive flux (power-law)"

        # Pixel 3: Both bounds zero, should be zero
        assert result_flux[0, 3] == 0.0, "Pixel 3 should be zero (both bounds zero)"


class TestGetPsetDirectionalMask:
    """Test suite for get_pset_direction_bin_mask function."""

    @pytest.fixture
    def pset_with_ram_mask(self):
        """Create a test dataset with ram_mask for testing."""
        n_epoch = 2
        n_spin = 100
        n_energy = 3

        # Create boolean mask with known pattern
        # First half True (ram), second half False (anti-ram)
        ram_mask_values = np.zeros((n_epoch, n_energy, n_spin), dtype=bool)
        ram_mask_values[:, :, :50] = True

        # Create hae_longitude with the same shape for epochs
        hae_lon_values = np.tile(
            np.linspace(0, 360, n_spin, endpoint=False), (n_epoch, 1)
        )

        dataset = xr.Dataset(
            {
                "epoch": (["epoch"], np.array([1e18, 1.1e18])),
                "ram_mask": (
                    ["epoch", "esa_energy_step", "spin_angle_bin"],
                    ram_mask_values,
                ),
                "hae_longitude": (
                    ["epoch", "spin_angle_bin"],
                    hae_lon_values,
                ),
            },
            coords={
                "esa_energy_step": np.arange(n_energy),
                "spin_angle_bin": np.arange(n_spin),
            },
        )

        return dataset

    def test_invalid_direction_raises_error(self, pset_with_ram_mask):
        """Test that invalid direction string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid direction string"):
            get_pset_directional_mask(pset_with_ram_mask, "invalid")

    def test_ram_direction_returns_ram_mask(self, pset_with_ram_mask):
        """Test that 'ram' direction returns the ram_mask."""
        result = get_pset_directional_mask(pset_with_ram_mask, "ram")

        # Should return the ram_mask DataArray
        assert isinstance(result, xr.DataArray)
        assert result.name == "ram_mask"

        # Values should match the original ram_mask
        np.testing.assert_array_equal(
            result.values, pset_with_ram_mask["ram_mask"].values
        )

    def test_anti_direction_returns_inverted_mask(self, pset_with_ram_mask):
        """Test that 'anti' direction returns inverted ram_mask."""
        result = get_pset_directional_mask(pset_with_ram_mask, "anti")

        # Should return a DataArray
        assert isinstance(result, xr.DataArray)

        # Values should be inverted from the original ram_mask
        expected = ~pset_with_ram_mask["ram_mask"]
        np.testing.assert_array_equal(result.values, expected.values)

    def test_full_direction_returns_none(self, pset_with_ram_mask):
        """Test that 'full' direction returns None."""
        result = get_pset_directional_mask(pset_with_ram_mask, "full")

        # Should return None
        assert result is None

    def test_ram_mask_dimensions_preserved(self, pset_with_ram_mask):
        """Test that returned mask has same dimensions as input ram_mask."""
        result = get_pset_directional_mask(pset_with_ram_mask, "ram")

        # Dimensions should match original ram_mask
        assert result.dims == pset_with_ram_mask["ram_mask"].dims
        assert result.shape == pset_with_ram_mask["ram_mask"].shape

    def test_anti_mask_dimensions_preserved(self, pset_with_ram_mask):
        """Test that anti-ram mask has same dimensions as input ram_mask."""
        result = get_pset_directional_mask(pset_with_ram_mask, "anti")

        # Dimensions should match original ram_mask
        assert result.dims == pset_with_ram_mask["ram_mask"].dims
        assert result.shape == pset_with_ram_mask["ram_mask"].shape

    def test_ram_mask_boolean_type(self, pset_with_ram_mask):
        """Test that returned masks are boolean type."""
        ram_result = get_pset_directional_mask(pset_with_ram_mask, "ram")
        anti_result = get_pset_directional_mask(pset_with_ram_mask, "anti")

        assert ram_result.dtype == bool
        assert anti_result.dtype == bool

    def test_ram_and_anti_are_complementary(self, pset_with_ram_mask):
        """Test that ram and anti masks are complementary."""
        ram_mask = get_pset_directional_mask(pset_with_ram_mask, "ram")
        anti_mask = get_pset_directional_mask(pset_with_ram_mask, "anti")

        # ram_mask and anti_mask should be complementary
        # (no overlap, cover all pixels)
        combined = ram_mask.values | anti_mask.values
        assert np.all(combined), "RAM and anti-RAM masks should cover all pixels"

        overlap = ram_mask.values & anti_mask.values
        assert not np.any(overlap), "RAM and anti-RAM masks should not overlap"

    def test_with_1d_spatial_dimension(self):
        """Test with 1D spatial dimension (like HiPointingSet)."""
        n_spin = 50

        # Create dataset with 1D spatial dimension
        ram_mask_values = np.zeros(n_spin, dtype=bool)
        ram_mask_values[:25] = True  # First half is ram

        dataset = xr.Dataset(
            {
                "ram_mask": (["spin_angle_bin"], ram_mask_values),
                "hae_longitude": (
                    ["spin_angle_bin"],
                    np.linspace(0, 360, n_spin, endpoint=False),
                ),
            },
            coords={"spin_angle_bin": np.arange(n_spin)},
        )

        ram_result = get_pset_directional_mask(dataset, "ram")
        anti_result = get_pset_directional_mask(dataset, "anti")

        # Verify shape is preserved
        assert ram_result.shape == (n_spin,)
        assert anti_result.shape == (n_spin,)

        # Verify values
        assert np.sum(ram_result.values) == 25
        assert np.sum(anti_result.values) == 25

    def test_with_2d_spatial_dimension(self):
        """Test with 2D spatial dimension (like LoPointingSet)."""
        n_spin = 20
        n_off = 10

        # Create dataset with 2D spatial dimensions
        ram_mask_values = np.zeros((n_spin, n_off), dtype=bool)
        ram_mask_values[:10, :] = True  # First half of spin dimension is ram

        dataset = xr.Dataset(
            {
                "ram_mask": (["spin_angle", "off_angle"], ram_mask_values),
                "hae_longitude": (
                    ["spin_angle", "off_angle"],
                    np.tile(
                        np.linspace(0, 360, n_spin, endpoint=False)[:, np.newaxis],
                        (1, n_off),
                    ),
                ),
            },
            coords={
                "spin_angle": np.arange(n_spin),
                "off_angle": np.arange(n_off),
            },
        )

        ram_result = get_pset_directional_mask(dataset, "ram")
        anti_result = get_pset_directional_mask(dataset, "anti")

        # Verify shape is preserved
        assert ram_result.shape == (n_spin, n_off)
        assert anti_result.shape == (n_spin, n_off)

        # Verify values
        assert np.sum(ram_result.values) == 10 * n_off
        assert np.sum(anti_result.values) == 10 * n_off
