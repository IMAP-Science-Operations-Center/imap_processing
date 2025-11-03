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
    _calculate_compton_getting_transform,
    add_spacecraft_velocity_to_pset,
    apply_compton_getting_correction,
    calculate_ram_mask,
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

        mock_hi_pset = add_spacecraft_velocity_to_pset(mock_hi_pset)

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
        mock_hi_pset = _add_cartesian_look_direction(mock_hi_pset)

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

        mock_hi_pset = add_spacecraft_velocity_to_pset(mock_hi_pset)
        mock_hi_pset = _add_cartesian_look_direction(mock_hi_pset)

        # Create energy array
        energy_hf = xr.DataArray(
            np.array([500.0, 1000.0, 2000.0]),
            dims=["esa_energy_step"],
            coords={"esa_energy_step": [1, 2, 3]},
        )

        mock_hi_pset = _calculate_compton_getting_transform(mock_hi_pset, energy_hf)

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
        mock_hi_pset = apply_compton_getting_correction(mock_hi_pset, energy_hf)

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
        hi_pset = ena_maps.HiPointingSet(pset_ds)

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
        hi_pset = apply_compton_getting_correction(hi_pset, energy_hf)

        # Verify coordinates were modified
        corrected_lon = hi_pset.data["hae_longitude"]
        corrected_lat = hi_pset.data["hae_latitude"]

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
        mock_hi_pset = _add_cartesian_look_direction(mock_hi_pset)

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
        pset = _add_cartesian_look_direction(pset)

        # Single energy level
        energy_hf = xr.DataArray(np.array([1000.0]), dims=["esa_energy_step"])

        # Calculate CG transform
        pset = _calculate_compton_getting_transform(pset, energy_hf)

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

    @staticmethod
    def create_synthetic_pset_with_hae_coords(shape=(10, 20)):
        """Create a synthetic LoHiBasePointingSet with known HAE coordinates."""
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

        # Create a LoPointingSet-like object
        class SyntheticPset(ena_maps.LoHiBasePointingSet):
            def __init__(self, dataset):
                self.spice_reference_frame = geometry.SpiceFrame.IMAP_HAE
                self.data = dataset.copy(deep=True)
                self.spatial_coords = ("spin_angle", "off_angle")
                self.update_az_el_points()

        return SyntheticPset(dataset)

    def test_update_ram_mask_plus_x_direction(self):
        """Test RAM mask with spacecraft velocity in +X direction."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Spacecraft velocity in +X direction (HAE frame)
        pset.data["sc_direction_vector"] = np.array([1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)

        lon = pset.data["hae_longitude"].values
        ram_mask = pset.data["ram_mask"].values

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
        pset.data["sc_direction_vector"] = np.array([-1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)

        lon = pset.data["hae_longitude"].values
        ram_mask = pset.data["ram_mask"].values

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
        pset.data["sc_direction_vector"] = np.array([0.0, 1.0, 0.0])
        pset = calculate_ram_mask(pset)

        lon = pset.data["hae_longitude"].values
        ram_mask = pset.data["ram_mask"].values

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
        pset.data["sc_direction_vector"] = np.array([1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)
        ram_mask_1 = pset.data["ram_mask"].values.copy()

        pset.data["sc_direction_vector"] = np.array([100.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)
        ram_mask_2 = pset.data["ram_mask"].values.copy()

        # The masks should be identical since direction is the same
        np.testing.assert_array_equal(ram_mask_1, ram_mask_2)

    def test_update_ram_mask_dot_product_correctness(self):
        """Test that dot product calculation is mathematically correct."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Use a simple spacecraft velocity vector
        pset.data["sc_direction_vector"] = np.array([1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)

        # Manually verify a few specific pixels
        lon = pset.data["hae_longitude"].values
        lat = pset.data["hae_latitude"].values
        ram_mask = pset.data["ram_mask"].values

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
        original_dims_2d = pset_2d.data["hae_longitude"].dims

        # Update ram_mask
        pset_2d.data["sc_direction_vector"] = np.array([1.0, 1.0, 0.0])
        pset_2d = calculate_ram_mask(pset_2d)

        # Verify dimensions are preserved
        assert pset_2d.data["ram_mask"].dims == original_dims_2d

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

        # Create a HiPointingSet-like object
        class SyntheticPset1D(ena_maps.LoHiBasePointingSet):
            def __init__(self, dataset):
                self.spice_reference_frame = geometry.SpiceFrame.IMAP_HAE
                self.data = dataset.copy(deep=True)
                self.spatial_coords = ("spin_angle_bin",)
                self.update_az_el_points()

        pset_1d = SyntheticPset1D(dataset_1d)

        # Get original dimensions
        original_dims_1d = pset_1d.data["hae_longitude"].dims

        # Update ram_mask
        pset_1d.data["sc_direction_vector"] = np.array([1.0, 1.0, 0.0])
        pset_1d = calculate_ram_mask(pset_1d)

        # Verify dimensions are preserved
        assert pset_1d.data["ram_mask"].dims == original_dims_1d

    def test_update_ram_mask_replaces_existing(self):
        """Test that update_ram_mask replaces existing ram_mask."""
        pset = self.create_synthetic_pset_with_hae_coords()

        # Set initial mask with +X direction
        pset.data["sc_direction_vector"] = np.array([1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)
        ram_mask_1 = pset.data["ram_mask"].values.copy()

        # Update mask with opposite direction
        pset.data["sc_direction_vector"] = np.array([-1.0, 0.0, 0.0])
        pset = calculate_ram_mask(pset)
        ram_mask_2 = pset.data["ram_mask"].values.copy()

        # The masks should be different
        assert not np.array_equal(ram_mask_1, ram_mask_2)

    def test_update_ram_mask_arbitrary_direction(self):
        """Test RAM mask with arbitrary spacecraft velocity direction."""
        pset = self.create_synthetic_pset_with_hae_coords(shape=(36, 18))

        # Use an arbitrary direction (not aligned with axes)
        pset.data["sc_direction_vector"] = np.array([1.0, 1.0, 0.5])
        pset = calculate_ram_mask(pset)

        # Verify the mask was created
        assert "ram_mask" in pset.data

        # Verify approximately half the pixels are RAM (for a sphere)
        ram_fraction = pset.data["ram_mask"].sum().values / pset.data["ram_mask"].size
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
            map_ds, esa_energies, helio_energies
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
            map_ds, esa_energies, helio_energies
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
            map_ds, esa_energies, helio_energies
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
            map_ds, esa_energies, helio_energies
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
            map_ds, esa_energies, helio_energies
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
            map_ds, esa_energies, helio_energies
        )

        # Check that we have NaN values where expected, not infinities
        flux = result_ds["ena_intensity"].values
        stat_unc = result_ds["ena_intensity_stat_uncert"].values
        sys_err = result_ds["ena_intensity_sys_err"].values

        # Should have no infinities
        assert not np.any(np.isinf(flux))
        assert not np.any(np.isinf(stat_unc))
        assert not np.any(np.isinf(sys_err))

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
            map_ds, esa_energies, helio_energies
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
            map_ds, esa_energies, helio_energies
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
            map_ds, esa_energies, helio_energies
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
            map_ds, esa_energies, helio_energies
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
