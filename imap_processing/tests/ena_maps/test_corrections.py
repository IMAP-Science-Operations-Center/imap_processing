"""Test coverage for ena_maps.corrections module."""

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.corrections import (
    PowerLawFluxCorrector,
    _add_cartesian_look_direction,
    _add_spacecraft_velocity_to_pset,
    _calculate_compton_getting_transform,
    apply_compton_getting_correction,
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


@pytest.fixture
def hi_pset_cdf_path(imap_tests_path):
    """Path to test Hi PSET CDF file."""
    return imap_tests_path / "hi/data/l1/imap_hi_l1c_45sensor-pset_20250415_v999.cdf"


@pytest.fixture
def mock_hi_pset():
    """Create a minimal mock Hi pointing set for testing."""
    # Create a simple dataset with necessary fields
    n_epoch = 1
    n_spin = 100

    data = xr.Dataset(
        {
            "epoch": (["epoch"], np.array([797949131184000000])),
            "hae_longitude": (
                ["epoch", "spin_angle_bin"],
                np.linspace(0, 360, n_spin, endpoint=False).reshape(n_epoch, n_spin),
            ),
            "hae_latitude": (
                ["epoch", "spin_angle_bin"],
                np.linspace(-90, 90, n_spin).reshape(n_epoch, n_spin),
            ),
            "spin_angle_bin": (["spin_angle_bin"], np.arange(n_spin)),
        }
    )

    # Create a mock pointing set object
    pset = mock.MagicMock(spec=ena_maps.LoHiBasePointingSet)
    pset.data = data
    pset.spatial_coords = ("spin_angle_bin",)
    pset.spice_reference_frame = geometry.SpiceFrame.IMAP_HAE

    return pset


class TestComptonGettingCorrection:
    """Test suite for Compton-Getting correction functions."""

    @mock.patch("imap_processing.ena_maps.utils.corrections.ttj2000ns_to_et")
    @mock.patch("imap_processing.ena_maps.utils.corrections.geometry.imap_state")
    def test_add_spacecraft_velocity_to_pset(
        self, mock_imap_state, mock_ttj2000_to_et, mock_hi_pset
    ):
        """Test that spacecraft velocity is correctly added to pointing set."""
        # Mock conversion from TTJ2000ns to ET
        et = 1000.0
        mock_ttj2000_to_et.return_value = et
        # Mock spacecraft state vector (position + velocity in HAE frame)
        mock_sc_state = np.array([1e8, 2e8, 3e8, 10.0, 20.0, 30.0])  # km and km/s
        mock_imap_state.return_value = mock_sc_state

        _add_spacecraft_velocity_to_pset(mock_hi_pset)

        # Verify SPICE was called correctly
        mock_imap_state.assert_called_once_with(
            et, ref_frame=geometry.SpiceFrame.IMAP_HAE
        )

        # Verify sc_velocity was added
        assert "sc_velocity" in mock_hi_pset.data
        assert isinstance(mock_hi_pset.data["sc_velocity"], xr.DataArray)
        np.testing.assert_array_equal(
            mock_hi_pset.data["sc_velocity"].values, np.array([10.0, 20.0, 30.0])
        )

        # Verify sc_direction_vector was added
        assert "sc_direction_vector" in mock_hi_pset.data
        expected_speed = np.sqrt(10**2 + 20**2 + 30**2)
        expected_direction = np.array([10.0, 20.0, 30.0]) / expected_speed
        np.testing.assert_allclose(
            mock_hi_pset.data["sc_direction_vector"].values, expected_direction
        )

    def test_add_cartesian_look_direction(self, mock_hi_pset):
        """Test that look directions are correctly calculated and added."""
        _add_cartesian_look_direction(mock_hi_pset)

        # _add_cartesian_look_direction is just a wrapper around
        # geometry.spherical_to_cartesian. We only need to test that the
        # look_direction variable was added and has the correct shape.
        # Verify look_direction was added
        assert "look_direction" in mock_hi_pset.data
        assert isinstance(mock_hi_pset.data["look_direction"], xr.DataArray)

        # Verify shape
        expected_shape = (1, 100, 3)  # (epoch, spin_angle_bin, x_y_z)
        assert mock_hi_pset.data["look_direction"].shape == expected_shape

    @mock.patch("imap_processing.ena_maps.utils.corrections.geometry.imap_state")
    def test_calculate_compton_getting_transform(self, mock_imap_state, mock_hi_pset):
        """Test Compton-Getting transformation calculations."""
        # Set up spacecraft velocity
        mock_sc_state = np.array([1e8, 2e8, 3e8, 10.0, 20.0, 30.0])
        mock_imap_state.return_value = mock_sc_state

        _add_spacecraft_velocity_to_pset(mock_hi_pset)
        _add_cartesian_look_direction(mock_hi_pset)

        # Create energy array
        energy_hf = xr.DataArray(
            np.array([500.0, 1000.0, 2000.0]),
            dims=["esa_energy_step"],
            coords={"esa_energy_step": [1, 2, 3]},
        )

        _calculate_compton_getting_transform(mock_hi_pset, energy_hf)

        # Verify required variables were added
        assert "energy_hf" in mock_hi_pset.data
        assert "energy_sc" in mock_hi_pset.data
        assert "hae_longitude" in mock_hi_pset.data
        assert "hae_latitude" in mock_hi_pset.data
        assert "ram_mask" in mock_hi_pset.data

        # Verify energy_hf matches input
        np.testing.assert_array_equal(
            mock_hi_pset.data["energy_hf"].values, energy_hf.values
        )

        # Verify energy_sc has correct shape
        # The transformation should broadcast across energy and spatial dimensions
        assert "energy_sc" in mock_hi_pset.data
        energy_sc = mock_hi_pset.data["energy_sc"]
        # Shape should include energy dimension
        assert len(energy_sc.dims) >= 2

        # Verify energy_sc values are positive and reasonable
        assert np.all(energy_sc.values > 0)
        assert np.all(np.isfinite(energy_sc.values))

        # Verify corrected coordinates are within valid ranges
        assert np.all(mock_hi_pset.data["hae_longitude"].values >= 0)
        assert np.all(mock_hi_pset.data["hae_longitude"].values <= 360)
        assert np.all(mock_hi_pset.data["hae_latitude"].values >= -90)
        assert np.all(mock_hi_pset.data["hae_latitude"].values <= 90)

        # Verify ram_mask properties
        ram_mask = mock_hi_pset.data["ram_mask"]
        assert isinstance(ram_mask, xr.DataArray)
        assert ram_mask.dtype == bool
        assert ram_mask.shape == mock_hi_pset.data["energy_sc"].shape

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

        # Apply the full correction
        apply_compton_getting_correction(mock_hi_pset, energy_hf)

        # Verify all intermediate variables were added
        assert "sc_velocity" in mock_hi_pset.data
        assert "sc_direction_vector" in mock_hi_pset.data
        assert "look_direction" in mock_hi_pset.data
        assert "energy_hf" in mock_hi_pset.data
        assert "energy_sc" in mock_hi_pset.data
        assert "hae_longitude" in mock_hi_pset.data
        assert "hae_latitude" in mock_hi_pset.data
        assert "ram_mask" in mock_hi_pset.data

        # Verify update_az_el_points was called
        mock_hi_pset.update_az_el_points.assert_called_once()

    @pytest.mark.external_test_data
    @mock.patch("imap_processing.ena_maps.utils.corrections.geometry.imap_state")
    def test_compton_getting_with_real_pset(self, mock_imap_state, hi_pset_cdf_path):
        """Test Compton-Getting correction with real Hi PSET data."""
        # Load real pointing set
        pset_ds = load_cdf(hi_pset_cdf_path)
        hi_pset = ena_maps.HiPointingSet(pset_ds, spin_phase="full")

        # Store original coordinates for comparison
        original_lon = hi_pset.data["hae_longitude"].copy()

        # Mock spacecraft state
        mock_sc_state = np.array([1e8, 2e8, 3e8, 12.0, -27.0, 0.02])  # km and km/s
        mock_imap_state.return_value = mock_sc_state

        # Create energy array (Hi has 9 energy steps)
        energy_hf = xr.DataArray(
            np.array(
                [500.0, 750.0, 1100.0, 1650.0, 2500.0, 3750.0, 5700.0, 8520.0, 12800.0]
            ),
            dims=["esa_energy_step"],
            coords={"esa_energy_step": np.arange(1, 10)},
        )

        # Apply correction
        apply_compton_getting_correction(hi_pset, energy_hf)

        # Verify coordinates were modified
        corrected_lon = hi_pset.data["hae_longitude"]
        corrected_lat = hi_pset.data["hae_latitude"]

        # Shape should now include energy dimension
        assert "esa_energy_step" in corrected_lon.dims
        assert "esa_energy_step" in corrected_lat.dims

        # Verify the correction changes the coordinates
        # (at least some points should be different)
        # Note: We can't directly compare because dimensions changed
        assert corrected_lon.shape != original_lon.shape

        # Verify all values are in valid ranges
        assert np.all(corrected_lon.values >= 0)
        assert np.all(corrected_lon.values <= 360)
        assert np.all(corrected_lat.values >= -90)
        assert np.all(corrected_lat.values <= 90)

        # Verify az_el_points was updated
        assert hi_pset.az_el_points is not None
        assert isinstance(hi_pset.az_el_points, xr.DataArray)

        # Verify ram_mask was added and has correct properties
        assert "ram_mask" in hi_pset.data
        ram_mask = hi_pset.data["ram_mask"]
        assert isinstance(ram_mask, xr.DataArray)
        assert ram_mask.dtype == bool
        assert ram_mask.shape == hi_pset.data["energy_sc"].shape

    def test_compton_getting_physical_consistency(self, mock_hi_pset):
        """Test physical consistency of Compton-Getting correction."""
        # Set up a known spacecraft velocity
        sc_velocity = np.array([30.0, 0.0, 0.0])  # Moving in +X direction at 30 km/s

        mock_hi_pset.data["sc_velocity"] = xr.DataArray(sc_velocity, dims=["x_y_z"])
        mock_hi_pset.data["sc_direction_vector"] = xr.DataArray(
            sc_velocity / np.linalg.norm(sc_velocity), dims=["x_y_z"]
        )

        # Set up simple look directions
        _add_cartesian_look_direction(mock_hi_pset)

        # Single energy level
        energy_hf = xr.DataArray(np.array([1000.0]), dims=["esa_energy_step"])

        _calculate_compton_getting_transform(mock_hi_pset, energy_hf)

        # Physical checks:
        # 1. Energy in spacecraft frame should be higher for particles coming
        #    from the direction of spacecraft motion (ram direction)
        energy_sc = mock_hi_pset.data["energy_sc"]

        # 2. All energies should be positive
        assert np.all(energy_sc.values > 0)

        # 3. The spacecraft frame energy should differ from heliosphere frame
        # (they should not all be exactly 1000 eV)
        assert not np.allclose(energy_sc.values, 1000.0)

        # 4. Energy variation should exist across different look directions
        assert energy_sc.values.std() > 0

    def test_ram_mask_calculation(self):
        """Test ram_mask correctly identifies ram and anti-ram directions."""
        # Create a simple mock pset with specific look directions
        n_directions = 4
        data = xr.Dataset(
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

        pset = mock.MagicMock(spec=ena_maps.LoHiBasePointingSet)
        pset.data = data
        pset.spatial_coords = ("direction",)
        pset.spice_reference_frame = geometry.SpiceFrame.IMAP_HAE

        # Set up spacecraft velocity in +X direction
        sc_velocity = np.array([30.0, 0.0, 0.0])  # km/s
        pset.data["sc_velocity"] = xr.DataArray(sc_velocity, dims=["x_y_z"])
        pset.data["sc_direction_vector"] = xr.DataArray(
            sc_velocity / np.linalg.norm(sc_velocity), dims=["x_y_z"]
        )

        # Add look directions
        _add_cartesian_look_direction(pset)

        # Single energy level
        energy_hf = xr.DataArray(np.array([1000.0]), dims=["esa_energy_step"])

        # Calculate CG transform
        _calculate_compton_getting_transform(pset, energy_hf)

        # Verify ram_mask exists
        assert "ram_mask" in pset.data
        ram_mask = pset.data["ram_mask"]

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
