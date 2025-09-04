"""Comprehensive test suite for IMAP-Lo L2 data processing."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo.l1c.lo_l1c import (
    ESA_ENERGY_STEPS,
    N_OFF_ANGLE_BINS,
    N_SPIN_ANGLE_BINS,
    OFF_ANGLE_BIN_CENTERS,
    PSET_DIMS,
    PSET_SHAPE,
    SPIN_ANGLE_BIN_CENTERS,
)
from imap_processing.lo.l2.lo_l2 import (
    add_efficiency_factors_to_pset,
    calculate_all_rates_and_intensities,
    calculate_backgrounds,
    calculate_efficiency_corrected_quantities,
    calculate_intensities,
    calculate_rates,
    cleanup_intermediate_variables,
    create_sky_map_from_psets,
    initialize_geometric_factor_variables,
    lo_l2,
    load_efficiency_data,
    normalize_pset_coordinates,
    populate_geometric_factors,
)

# =============================================================================
# FIXTURES FOR MOCK DATA
# =============================================================================


@pytest.fixture
def sample_pset():
    """Create a sample pointing set with typical data variables."""
    # Create counts data with some non-zero values
    h_counts = np.zeros(PSET_SHAPE)
    h_counts[:, 2:4, 10:20, 5:15] = 5  # Add some counts for testing

    o_counts = np.zeros(PSET_SHAPE)
    o_counts[:, 1:3, 15:25, 8:18] = 3

    doubles_counts = np.zeros(PSET_SHAPE)
    doubles_counts[:, 0:2, 5:15, 10:20] = 2

    triples_counts = np.zeros(PSET_SHAPE)
    triples_counts[:, 3:5, 20:30, 15:25] = 1

    exposure_time = np.full(PSET_SHAPE, 0.5)

    # Create background rates data for h and o only
    h_background_rates = np.full(PSET_SHAPE, 0.1)  # 0.1 counts/s background
    o_background_rates = np.full(PSET_SHAPE, 0.05)  # 0.05 counts/s background
    h_background_rates_stat_uncert = np.full(PSET_SHAPE, 0.01)  # 10% uncertainty
    o_background_rates_stat_uncert = np.full(PSET_SHAPE, 0.005)  # 10% uncertainty

    # Create coordinate arrays
    lons, lats = np.meshgrid(
        SPIN_ANGLE_BIN_CENTERS, OFF_ANGLE_BIN_CENTERS, indexing="ij"
    )
    hae_longitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_latitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_longitude[0, :, :] = lons
    hae_latitude[0, :, :] = lats

    dataset = xr.Dataset(
        {
            "h_counts": (PSET_DIMS, h_counts),
            "o_counts": (PSET_DIMS, o_counts),
            "doubles_counts": (PSET_DIMS, doubles_counts),
            "triples_counts": (PSET_DIMS, triples_counts),
            "exposure_time": (PSET_DIMS, exposure_time),
            "h_bg_rate": (PSET_DIMS, h_background_rates),
            "o_bg_rate": (PSET_DIMS, o_background_rates),
            "h_bg_rate_stat_uncert": (
                PSET_DIMS,
                h_background_rates_stat_uncert,
            ),
            "o_bg_rate_stat_uncert": (
                PSET_DIMS,
                o_background_rates_stat_uncert,
            ),
            "hae_longitude": (("epoch", "spin_angle", "off_angle"), hae_longitude),
            "hae_latitude": (("epoch", "spin_angle", "off_angle"), hae_latitude),
        },
        coords={
            "epoch": [8.1794907049e17],
            "esa_energy_step": ESA_ENERGY_STEPS,
            "spin_angle": SPIN_ANGLE_BIN_CENTERS,
            "off_angle": OFF_ANGLE_BIN_CENTERS,
        },
    )
    return dataset


@pytest.fixture
def minimal_pset():
    """Create a minimal pointing set with all count types for testing."""
    h_counts = np.ones(PSET_SHAPE)  # All ones for easy testing
    o_counts = np.ones(PSET_SHAPE) * 0.5  # Half the hydrogen counts
    doubles_counts = np.ones(PSET_SHAPE) * 0.2  # Some doubles events
    triples_counts = np.ones(PSET_SHAPE) * 0.1  # Some triples events
    exposure_time = np.full(PSET_SHAPE, 1.0)  # 1 second exposure for easy math

    # Create simple background rates for testing
    h_background_rates = np.full(PSET_SHAPE, 0.2)  # 0.2 counts/s
    o_background_rates = np.full(PSET_SHAPE, 0.1)  # 0.1 counts/s
    h_background_rates_stat_uncert = np.full(PSET_SHAPE, 0.02)  # 10% uncertainty
    o_background_rates_stat_uncert = np.full(PSET_SHAPE, 0.01)  # 10% uncertainty

    # Simple coordinate arrays
    lons, lats = np.meshgrid(
        SPIN_ANGLE_BIN_CENTERS, OFF_ANGLE_BIN_CENTERS, indexing="ij"
    )
    hae_longitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_latitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_longitude[0, :, :] = lons
    hae_latitude[0, :, :] = lats

    dataset = xr.Dataset(
        {
            "h_counts": (PSET_DIMS, h_counts),
            "o_counts": (PSET_DIMS, o_counts),
            "doubles_counts": (PSET_DIMS, doubles_counts),
            "triples_counts": (PSET_DIMS, triples_counts),
            "exposure_time": (PSET_DIMS, exposure_time),
            "h_background_rates": (PSET_DIMS, h_background_rates),
            "o_background_rates": (PSET_DIMS, o_background_rates),
            "h_background_rates_stat_uncert": (
                PSET_DIMS,
                h_background_rates_stat_uncert,
            ),
            "o_background_rates_stat_uncert": (
                PSET_DIMS,
                o_background_rates_stat_uncert,
            ),
            "hae_longitude": (("epoch", "spin_angle", "off_angle"), hae_longitude),
            "hae_latitude": (("epoch", "spin_angle", "off_angle"), hae_latitude),
        },
        coords={
            "epoch": [8.1794907049e17],
            "esa_energy_step": ESA_ENERGY_STEPS,
            "spin_angle": SPIN_ANGLE_BIN_CENTERS,
            "off_angle": OFF_ANGLE_BIN_CENTERS,
        },
    )
    return dataset


@pytest.fixture
def sample_efficiency_data():
    """Create sample efficiency factor data for testing."""
    data = {
        "Date": [np.datetime64("2025-01-01"), np.datetime64("2025-01-02")],
        "E-Step1_eff": [0.8, 0.85],
        "E-Step2_eff": [0.82, 0.87],
        "E-Step3_eff": [0.84, 0.89],
        "E-Step4_eff": [0.86, 0.91],
        "E-Step5_eff": [0.88, 0.93],
        "E-Step6_eff": [0.90, 0.95],
        "E-Step7_eff": [0.92, 0.97],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_geometric_factor_data():
    """Create sample geometric factor data for testing."""
    h_gf_data = []
    o_gf_data = []

    for i in range(7):  # 7 energy steps
        h_gf_data.append(
            {
                "esa_mode": 0,
                "Observed_E-Step": i + 1,
                "Cntr_E": 0.01 * (i + 1),  # Simple energy values
                "Cntr_E_unc": 0.001 * (i + 1),
                "GF_Trpl_H": 1e-4 * (i + 1),
                "GF_Trpl_H_unc": 1e-5 * (i + 1),
                "GF_Dbl_all": 2e-4 * (i + 1),
                "GF_Dbl_all_unc": 2e-5 * (i + 1),
                "GF_Trpl_all": 3e-4 * (i + 1),
                "GF_Trpl_all_unc": 3e-5 * (i + 1),
            }
        )

        o_gf_data.append(
            {
                "esa_mode": 0,
                "Observed_E-Step": i + 1,
                "Cntr_E": 0.015 * (i + 1),  # Slightly different for oxygen
                "Cntr_E_unc": 0.0015 * (i + 1),
                "GF_Trpl_O": 1.5e-4 * (i + 1),
                "GF_Trpl_O_unc": 1.5e-5 * (i + 1),
            }
        )

    return pd.DataFrame(h_gf_data), pd.DataFrame(o_gf_data)


@pytest.fixture
def sample_sky_map_dataset():
    """Create a sample sky map dataset for testing calculations."""
    # Create a simple rectangular map
    n_lon, n_lat = 60, 30
    n_energy = 7

    dataset = xr.Dataset(
        coords={
            "epoch": [8.1794907049e17],
            "energy": list(range(n_energy)),
            "longitude": np.linspace(0, 360, n_lon, endpoint=False),
            "latitude": np.linspace(-90, 90, n_lat),
        }
    )

    # Add count data
    for var in ["h", "o", "doubles", "triples"]:
        counts = np.ones((1, n_energy, n_lon, n_lat)) * 10  # 10 counts for easy math
        dataset[f"{var}_counts"] = (
            ("epoch", "energy", "longitude", "latitude"),
            counts,
        )

        # Add efficiency-corrected quantities for intensity calculations
        eff_corr = counts / 0.9  # Assuming 90% efficiency
        dataset[f"{var}_counts_over_eff"] = (
            ("epoch", "energy", "longitude", "latitude"),
            eff_corr,
        )
        dataset[f"{var}_counts_over_eff_squared"] = (
            ("epoch", "energy", "longitude", "latitude"),
            eff_corr,
        )

    # Add exposure time
    exposure = np.ones((1, n_energy, n_lon, n_lat)) * 1.0  # 1 second
    dataset["exposure_time"] = (("epoch", "energy", "longitude", "latitude"), exposure)

    return dataset


@pytest.fixture
def sample_dataset_with_background_intermediates():
    """Create a dataset with background intermediate variables for testing."""
    # Create a simple rectangular map with background data
    n_energy = 7

    dataset = xr.Dataset(
        coords={
            "epoch": [8.1794907049e17],
            "energy": list(range(n_energy)),
        }
    )

    # Add the intermediate background variables that would be created
    # during projection from pset to map
    for var in ["h", "o"]:
        # Background rate data (already projected)
        bg_rate_exposure_time = np.ones((1, n_energy)) * 0.2  # 0.2 counts
        dataset[f"{var}_bg_rate_exposure_time"] = (
            ("epoch", "energy"),
            bg_rate_exposure_time,
        )

        # Background uncertainty squared times exposure time squared
        bg_rate_stat_uncert_exposure_time2 = np.ones((1, n_energy)) * 0.004  # 0.02^2
        dataset[f"{var}_bg_rate_stat_uncert_exposure_time2"] = (
            ("epoch", "energy"),
            bg_rate_stat_uncert_exposure_time2,
        )

    # Add exposure time (this would be the projected exposure time)
    exposure = np.ones((1, n_energy)) * 1.0  # 1 second
    dataset["exposure_time"] = (("epoch", "energy"), exposure)

    # Add geometric factors for systematic uncertainty calculation
    for var in ["h", "o"]:
        dataset[f"{var}_gf"] = (("energy",), np.ones(n_energy) * 1e-4)
        dataset[f"{var}_gf_stat_uncert"] = (("energy",), np.ones(n_energy) * 1e-5)

    return dataset


# =============================================================================
# UNIT TESTS FOR INDIVIDUAL FUNCTIONS
# =============================================================================


class TestLoadEfficiencyData:
    """Tests for the load_efficiency_data function."""

    def test_load_efficiency_data_with_files(self, tmp_path):
        """Test loading efficiency data when files are present."""
        # Create temporary efficiency files
        eff_file1 = tmp_path / "efficiency-factor_v001.csv"
        eff_file2 = tmp_path / "efficiency-factor_v002.csv"

        # Create sample data
        data1 = pd.DataFrame(
            {
                "Date": [np.datetime64("2025-01-01")],
                "E-Step1_eff": [0.8],
                "E-Step2_eff": [0.82],
                "E-Step3_eff": [0.84],
                "E-Step4_eff": [0.86],
                "E-Step5_eff": [0.88],
                "E-Step6_eff": [0.90],
                "E-Step7_eff": [0.92],
            }
        )

        data2 = pd.DataFrame(
            {
                "Date": [np.datetime64("2025-01-02")],
                "E-Step1_eff": [0.85],
                "E-Step2_eff": [0.87],
                "E-Step3_eff": [0.89],
                "E-Step4_eff": [0.91],
                "E-Step5_eff": [0.93],
                "E-Step6_eff": [0.95],
                "E-Step7_eff": [0.97],
            }
        )

        # Save to CSV
        data1.to_csv(eff_file1, index=False)
        data2.to_csv(eff_file2, index=False)

        # Mock the ancillary file reader
        with patch(
            "imap_processing.lo.l2.lo_l2.lo_ancillary.read_ancillary_file"
        ) as mock_read:
            mock_read.side_effect = [data1, data2]

            # Test the function
            result = load_efficiency_data([str(eff_file1), str(eff_file2)])

            # Verify results
            assert len(result) == 2
            assert "Date" in result.columns
            assert "E-Step1_eff" in result.columns
            assert mock_read.call_count == 2

    def test_load_efficiency_data_no_files(self):
        """Test loading efficiency data when no files are present."""
        result = load_efficiency_data([])

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_load_efficiency_data_non_efficiency_files(self):
        """Test that non-efficiency files are ignored."""
        files = ["some_other_file.csv", "another_file.txt"]
        result = load_efficiency_data(files)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestNormalizePsetCoordinates:
    """Tests for the normalize_pset_coordinates function."""

    def test_normalize_coordinates_basic(self, minimal_pset):
        """Test basic coordinate normalization."""
        # Create a mock output map with energy dimension and coordinates
        mock_output_map = Mock()
        mock_output_map.data_1d.dims = ["energy"]
        # Create energy coordinates that will be assigned
        energy_coords = np.array([10, 20, 30, 40, 50, 60, 70], dtype=float)
        mock_output_map.data_1d.coords.get.return_value = energy_coords

        result = normalize_pset_coordinates(minimal_pset, mock_output_map)

        # Check that dimensions were renamed
        assert "energy" in result.dims
        assert "esa_energy_step" not in result.dims

        # Check that energy coordinate is present and properly assigned
        assert "energy" in result.coords
        np.testing.assert_array_equal(result.coords["energy"], energy_coords)

        # Check that old coordinate variable was dropped
        assert "esa_energy_step" not in result.variables

    def test_normalize_coordinates_no_energy_in_map(self, minimal_pset):
        """Test normalization when output map has no energy dimension."""
        mock_output_map = Mock()
        mock_output_map.data_1d.dims = []

        result = normalize_pset_coordinates(minimal_pset, mock_output_map)

        # Should still rename dimensions
        assert "energy" in result.dims
        assert "esa_energy_step" not in result.dims

    def test_normalize_coordinates_removes_old_coordinate(self, minimal_pset):
        """Test that old esa_energy_step coordinate is removed."""
        # Add esa_energy_step as a variable (not just coordinate)
        pset_with_var = minimal_pset.copy()
        pset_with_var["esa_energy_step"] = xr.DataArray([1, 2, 3, 4, 5, 6, 7])

        mock_output_map = Mock()
        mock_output_map.data_1d.dims = []

        result = normalize_pset_coordinates(pset_with_var, mock_output_map)

        # Should remove the esa_energy_step variable
        assert "esa_energy_step" not in result.variables


class TestAddEfficiencyFactorsToPset:
    """Tests for the add_efficiency_factors_to_pset function."""

    def test_add_efficiency_factors_with_data(
        self, minimal_pset, sample_efficiency_data
    ):
        """Test adding efficiency factors when data is available."""
        # Set the epoch to match our sample data
        pset = minimal_pset.copy()
        # Convert date to TT2000 nanoseconds (approximate)
        epoch_ns = 8.1794907049e17  # This should correspond to 2025-01-01
        pset = pset.assign_coords(epoch=[epoch_ns])

        with (
            patch("imap_processing.lo.l2.lo_l2.ttj2000ns_to_et") as mock_ttj2000_to_et,
            patch("imap_processing.lo.l2.lo_l2.et_to_datetime64") as mock_et_to_dt64,
        ):
            # Mock the time conversion
            mock_ttj2000_to_et.return_value = 1234567890.0
            mock_et_to_dt64.return_value = np.datetime64("2025-01-01")

            result = add_efficiency_factors_to_pset(pset, sample_efficiency_data)

            # Check that efficiency was added
            assert "efficiency" in result.data_vars
            assert result["efficiency"].dims == ("energy",)
            assert len(result["efficiency"]) == 7

            # Check efficiency values match expected (first row of sample data)
            expected_eff = [0.8, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92]
            np.testing.assert_array_almost_equal(
                result["efficiency"].values, expected_eff
            )

    def test_add_efficiency_factors_no_data(self, minimal_pset):
        """Test adding efficiency factors when no data is available."""
        empty_df = pd.DataFrame()

        result = add_efficiency_factors_to_pset(minimal_pset, empty_df)

        # Should create unity efficiency
        assert "efficiency" in result.data_vars
        np.testing.assert_array_equal(result["efficiency"].values, np.ones(7))

    def test_add_efficiency_factors_missing_date(
        self, minimal_pset, sample_efficiency_data
    ):
        """Test error when efficiency factor not found for date."""
        pset = minimal_pset.copy()

        with (
            patch("imap_processing.lo.l2.lo_l2.ttj2000ns_to_et") as mock_ttj2000_to_et,
            patch("imap_processing.lo.l2.lo_l2.et_to_datetime64") as mock_et_to_dt64,
        ):
            # Mock conversion to a date not in sample data
            mock_ttj2000_to_et.return_value = 1234567890.0
            mock_et_to_dt64.return_value = np.datetime64("2025-12-31")

            with pytest.raises(ValueError, match="No efficiency factor found"):
                add_efficiency_factors_to_pset(pset, sample_efficiency_data)


class TestCalculateEfficiencyCorrectedQuantities:
    """Tests for the calculate_efficiency_corrected_quantities function."""

    def test_calculate_efficiency_corrected_quantities(self, sample_pset):
        """Test calculation of efficiency-corrected quantities."""
        # Add efficiency factors using the correct dimension name
        pset = sample_pset.copy()
        efficiency = np.array([0.8, 0.85, 0.9, 0.95, 0.88, 0.92, 0.87])
        pset["efficiency"] = xr.DataArray(efficiency, dims=["esa_energy_step"])

        result = calculate_efficiency_corrected_quantities(pset)

        # Check that corrected quantities were added
        for var in ["h", "o", "doubles", "triples"]:
            assert f"{var}_counts_over_eff" in result.data_vars
            assert f"{var}_counts_over_eff_squared" in result.data_vars

            # Check dimensions
            assert result[f"{var}_counts_over_eff"].dims == pset[f"{var}_counts"].dims

            # Check that division by efficiency happened
            expected_over_eff = pset[f"{var}_counts"] / pset["efficiency"]
            xr.testing.assert_allclose(
                result[f"{var}_counts_over_eff"], expected_over_eff
            )

            # Check that division by efficiency squared happened
            expected_over_eff_sq = pset[f"{var}_counts"] / (pset["efficiency"] ** 2)
            xr.testing.assert_allclose(
                result[f"{var}_counts_over_eff_squared"], expected_over_eff_sq
            )


class TestCalculateRates:
    """Tests for the calculate_rates function."""

    def test_calculate_rates_all_variables(self, sample_sky_map_dataset):
        """Test rate calculation for all particle types."""
        result = calculate_rates(sample_sky_map_dataset)

        # Check that rates were calculated for all variables
        for var in ["h", "o", "doubles", "triples"]:
            assert f"{var}_rate" in result.data_vars
            assert f"{var}_rate_stat_uncert" in result.data_vars

            # Check dimensions
            assert (
                result[f"{var}_rate"].dims
                == sample_sky_map_dataset[f"{var}_counts"].dims
            )

            # Check rate calculation (counts / exposure_time)
            # With counts=10 and exposure=1, rate should be 10
            assert np.all(result[f"{var}_rate"].values == 10.0)

            # Check uncertainty calculation (sqrt(counts) / exposure_time)
            # With counts=10 and exposure=1, uncertainty should be sqrt(10)
            expected_uncert = np.sqrt(10.0)
            assert np.allclose(
                result[f"{var}_rate_stat_uncert"].values, expected_uncert
            )

    def test_calculate_rates_missing_variables(self):
        """Rate calculation when some variables are missing should raise error."""
        # Create dataset with only hydrogen counts
        dataset = xr.Dataset(
            {
                "h_counts": (("epoch", "energy"), np.ones((1, 7)) * 5),
                "exposure_time": (("epoch", "energy"), np.ones((1, 7))),
            }
        )

        # The current function tries to access all variables,
        # so it should raise KeyError
        with pytest.raises(KeyError, match="No variable named 'o_counts'"):
            calculate_rates(dataset)


class TestCalculateIntensities:
    """Tests for the calculate_intensities function."""

    def test_calculate_intensities_h_and_o(self):
        """Test intensity calculation for hydrogen and oxygen."""
        # Create a dataset with the required variables
        dataset = xr.Dataset(
            {
                "h_counts_over_eff": (
                    ("energy",),
                    np.ones(7) * 100,
                ),  # 100 corrected counts
                "h_counts_over_eff_squared": (("energy",), np.ones(7) * 100),
                "o_counts_over_eff": (
                    ("energy",),
                    np.ones(7) * 50,
                ),  # 50 corrected counts
                "o_counts_over_eff_squared": (("energy",), np.ones(7) * 50),
                "exposure_time": (("energy",), np.ones(7) * 1.0),  # 1 second exposure
                "h_gf": (("energy",), np.ones(7) * 1e-4),  # Geometric factor
                "o_gf": (("energy",), np.ones(7) * 1e-4),
                "energy_h": (("energy",), np.ones(7) * 0.1),  # 0.1 keV
                "energy_o": (("energy",), np.ones(7) * 0.1),
                "h_gf_stat_uncert": (("energy",), np.ones(7) * 1e-5),  # 10% uncertainty
                "o_gf_stat_uncert": (("energy",), np.ones(7) * 1e-5),
            }
        )

        result = calculate_intensities(dataset)

        # Check that intensities were calculated
        for var in ["h", "o"]:
            assert f"{var}_intensity" in result.data_vars
            assert f"{var}_intensity_stat_uncert" in result.data_vars
            assert f"{var}_intensity_sys_err" in result.data_vars

        # Check intensity calculation:
        # counts_over_eff / (gf * energy * exposure_time)
        # For h: 100 / (1e-4 * 0.1 * 1.0) = 100 / 1e-5 = 1e7
        expected_h_intensity = 100 / (1e-4 * 0.1 * 1.0)
        assert np.allclose(result["h_intensity"].values, expected_h_intensity)

        # For o: 50 / (1e-4 * 0.1 * 1.0) = 50 / 1e-5 = 5e6
        expected_o_intensity = 50 / (1e-4 * 0.1 * 1.0)
        assert np.allclose(result["o_intensity"].values, expected_o_intensity)

    def test_calculate_intensities_missing_variables(self):
        """Test intensity calculation when some variables are missing."""
        # Create dataset with only hydrogen variables
        dataset = xr.Dataset(
            {
                "h_counts_over_eff": (("energy",), np.ones(7) * 100),
                "h_counts_over_eff_squared": (("energy",), np.ones(7) * 100),
                "exposure_time": (("energy",), np.ones(7) * 1.0),
                "h_gf": (("energy",), np.ones(7) * 1e-4),
                "energy_h": (("energy",), np.ones(7) * 0.1),
                "h_gf_stat_uncert": (("energy",), np.ones(7) * 1e-5),
            }
        )

        # Function should fail when trying to access missing 'o' variables
        with pytest.raises(KeyError, match="No variable named 'o_counts_over_eff'"):
            calculate_intensities(dataset)


class TestCalculateBackgrounds:
    """Tests for the calculate_backgrounds function."""

    def test_calculate_backgrounds_basic(
        self, sample_dataset_with_background_intermediates
    ):
        """Test basic background calculations with standard data."""
        dataset = sample_dataset_with_background_intermediates

        result = calculate_backgrounds(dataset)

        # Check that background intensities were calculated
        for var in ["h", "o"]:
            assert f"{var}_bg_intensity" in result.data_vars
            assert f"{var}_bg_intensity_stat_uncert" in result.data_vars
            assert f"{var}_bg_intensity_sys_err" in result.data_vars

        # Check background intensity calculation
        # bg_rate_exposure_time / exposure_time = 0.2 / 1.0 = 0.2
        expected_bg_intensity = 0.2
        assert np.allclose(result["h_bg_intensity"].values, expected_bg_intensity)
        assert np.allclose(result["o_bg_intensity"].values, expected_bg_intensity)

        # Check statistical uncertainty calculation
        # sqrt(bg_rate_stat_uncert_exposure_time2) / exposure_time
        # sqrt(0.004) / 1.0 = 0.063...
        expected_stat_uncert = np.sqrt(0.004) / 1.0
        assert np.allclose(
            result["h_bg_intensity_stat_uncert"].values, expected_stat_uncert
        )
        assert np.allclose(
            result["o_bg_intensity_stat_uncert"].values, expected_stat_uncert
        )

        # Check systematic uncertainty calculation
        # (gf_stat_uncert / gf) * bg_intensity = (1e-5 / 1e-4) * 0.2 = 0.02
        expected_sys_err = (1e-5 / 1e-4) * 0.2
        assert np.allclose(result["h_bg_intensity_sys_err"].values, expected_sys_err)
        assert np.allclose(result["o_bg_intensity_sys_err"].values, expected_sys_err)

    def test_calculate_backgrounds_zero_exposure(self):
        """Test background calculations with zero exposure time."""
        dataset = xr.Dataset(
            {
                "h_bg_rate_exposure_time": (("epoch", "energy"), np.ones((1, 7)) * 0.2),
                "o_bg_rate_exposure_time": (("epoch", "energy"), np.ones((1, 7)) * 0.1),
                "h_bg_rate_stat_uncert_exposure_time2": (
                    ("epoch", "energy"),
                    np.ones((1, 7)) * 0.004,
                ),
                "o_bg_rate_stat_uncert_exposure_time2": (
                    ("epoch", "energy"),
                    np.ones((1, 7)) * 0.001,
                ),
                "exposure_time": (
                    ("epoch", "energy"),
                    np.zeros((1, 7)),
                ),  # Zero exposure
                "h_gf": (("energy",), np.ones(7) * 1e-4),
                "o_gf": (("energy",), np.ones(7) * 1e-4),
                "h_gf_stat_uncert": (("energy",), np.ones(7) * 1e-5),
                "o_gf_stat_uncert": (("energy",), np.ones(7) * 1e-5),
            },
            coords={"epoch": [8.1794907049e17], "energy": list(range(7))},
        )

        result = calculate_backgrounds(dataset)

        # Should handle division by zero gracefully
        assert "h_bg_intensity" in result.data_vars
        assert "o_bg_intensity" in result.data_vars
        # Results should be infinite where exposure time is zero
        assert np.all(np.isinf(result["h_bg_intensity"].values))
        assert np.all(np.isinf(result["o_bg_intensity"].values))


class TestInitializeGeometricFactorVariables:
    """Tests for the initialize_geometric_factor_variables function."""

    def test_initialize_geometric_factor_variables(self):
        """Test initialization of geometric factor variables."""
        # Create a simple dataset
        dataset = xr.Dataset(
            {
                "test_var": (("energy",), np.ones(7)),
            },
            coords={"energy": range(7)},
        )

        result = initialize_geometric_factor_variables(dataset)

        # Check that all geometric factor variables were initialized
        expected_vars = [
            "energy_h",
            "energy_h_stat_uncert",
            "h_gf",
            "h_gf_stat_uncert",
            "energy_o",
            "energy_o_stat_uncert",
            "o_gf",
            "o_gf_stat_uncert",
            "doubles_gf",
            "doubles_gf_stat_uncert",
            "triples_gf",
            "triples_gf_stat_uncert",
        ]

        for var in expected_vars:
            assert var in result.data_vars
            assert result[var].dims == ("energy",)
            assert result[var].shape == (7,)
            assert np.all(result[var].values == 0)  # Should be initialized to zeros


class TestPopulateGeometricFactors:
    """Tests for the populate_geometric_factors function."""

    def test_populate_geometric_factors(self, sample_geometric_factor_data):
        """Test population of geometric factor values."""
        h_gf_data, o_gf_data = sample_geometric_factor_data

        # Create initialized dataset
        dataset = xr.Dataset(coords={"energy": range(7)})
        dataset = initialize_geometric_factor_variables(dataset)

        result = populate_geometric_factors(dataset, h_gf_data, o_gf_data)

        # Check that values were populated correctly
        for i in range(7):
            # Check hydrogen values
            assert result["energy_h"].values[i] == 0.01 * (i + 1)
            assert result["h_gf"].values[i] == 1e-4 * (i + 1)

            # Check oxygen values
            assert result["energy_o"].values[i] == 0.015 * (i + 1)
            assert result["o_gf"].values[i] == 1.5e-4 * (i + 1)

            # Check general geometric factors
            assert result["doubles_gf"].values[i] == 2e-4 * (i + 1)
            assert result["triples_gf"].values[i] == 3e-4 * (i + 1)


class TestCleanupIntermediateVariables:
    """Tests for the cleanup_intermediate_variables function."""

    def test_cleanup_intermediate_variables(self):
        """Test removal of intermediate variables."""
        # Create dataset with intermediate variables
        dataset = xr.Dataset(
            {
                "h_counts": (("energy",), np.ones(7)),
                "h_counts_over_eff": (("energy",), np.ones(7)),
                "h_counts_over_eff_squared": (("energy",), np.ones(7)),
                "h_gf": (("energy",), np.ones(7)),
                "h_gf_stat_uncert": (("energy",), np.ones(7)),
                "o_counts_over_eff": (("energy",), np.ones(7)),
                "o_gf": (("energy",), np.ones(7)),
                "h_bg_rate_exposure_time": (("energy",), np.ones(7)),
                "o_bg_rate_exposure_time": (("energy",), np.ones(7)),
                "h_bg_rate_stat_uncert_exposure_time2": (("energy",), np.ones(7)),
                "o_bg_rate_stat_uncert_exposure_time2": (("energy",), np.ones(7)),
                "h_intensity": (("energy",), np.ones(7)),  # Should be kept
                "exposure_time": (("energy",), np.ones(7)),  # Should be kept
            }
        )

        result = cleanup_intermediate_variables(dataset)

        # Should keep these variables
        assert "h_counts" in result.data_vars
        assert "h_intensity" in result.data_vars
        assert "exposure_time" in result.data_vars

        # Should remove these intermediate variables
        assert "h_counts_over_eff" not in result.data_vars
        assert "h_counts_over_eff_squared" not in result.data_vars
        assert "h_gf" not in result.data_vars
        assert "h_gf_stat_uncert" not in result.data_vars
        assert "o_counts_over_eff" not in result.data_vars
        assert "o_gf" not in result.data_vars
        assert "h_bg_rate_exposure_time" not in result.data_vars
        assert "o_bg_rate_exposure_time" not in result.data_vars
        assert "h_bg_rate_stat_uncert_exposure_time2" not in result.data_vars
        assert "o_bg_rate_stat_uncert_exposure_time2" not in result.data_vars

    def test_cleanup_partial_variables(self):
        """Test cleanup when only some intermediate variables exist."""
        # Create dataset with only some intermediate variables
        dataset = xr.Dataset(
            {
                "h_counts": (("energy",), np.ones(7)),
                "h_counts_over_eff": (("energy",), np.ones(7)),
                "exposure_time": (("energy",), np.ones(7)),
                # Missing: h_counts_over_eff_squared, h_gf, etc.
            }
        )

        result = cleanup_intermediate_variables(dataset)

        # Should keep these
        assert "h_counts" in result.data_vars
        assert "exposure_time" in result.data_vars

        # Should remove only the existing intermediate variable
        assert "h_counts_over_eff" not in result.data_vars


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCalculateAllRatesAndIntensities:
    """Integration tests for the calculate_all_rates_and_intensities function."""

    def test_calculate_all_rates_and_intensities_complete(self):
        """Test the complete rates and intensities calculation pipeline."""
        # Create a comprehensive dataset
        dataset = xr.Dataset(
            {
                # Count data (all required by calculate_rates)
                "h_counts": (("energy",), np.ones(7) * 10),
                "o_counts": (("energy",), np.ones(7) * 5),
                "doubles_counts": (("energy",), np.ones(7) * 2),
                "triples_counts": (("energy",), np.ones(7) * 1),
                # Efficiency corrected data
                "h_counts_over_eff": (("energy",), np.ones(7) * 12),  # 10/0.83 ≈ 12
                "h_counts_over_eff_squared": (("energy",), np.ones(7) * 12),
                "o_counts_over_eff": (("energy",), np.ones(7) * 6),  # 5/0.83 ≈ 6
                "o_counts_over_eff_squared": (("energy",), np.ones(7) * 6),
                # Other required data
                "exposure_time": (("energy",), np.ones(7) * 1.0),
                "h_gf": (("energy",), np.ones(7) * 1e-4),
                "o_gf": (("energy",), np.ones(7) * 1e-4),
                "energy_h": (("energy",), np.ones(7) * 0.1),
                "energy_o": (("energy",), np.ones(7) * 0.1),
                "h_gf_stat_uncert": (("energy",), np.ones(7) * 1e-5),
                "o_gf_stat_uncert": (("energy",), np.ones(7) * 1e-5),
                # Background intermediate data
                "h_bg_rate_exposure_time": (("energy",), np.ones(7) * 0.3),
                "o_bg_rate_exposure_time": (("energy",), np.ones(7) * 0.15),
                "h_bg_rate_stat_uncert_exposure_time2": (
                    ("energy",),
                    np.ones(7) * 0.009,
                ),
                "o_bg_rate_stat_uncert_exposure_time2": (
                    ("energy",),
                    np.ones(7) * 0.0025,
                ),
            }
        )

        result = calculate_all_rates_and_intensities(dataset)

        # Check that rates were calculated
        assert "h_rate" in result.data_vars
        assert "o_rate" in result.data_vars
        assert "doubles_rate" in result.data_vars
        assert "triples_rate" in result.data_vars
        assert "h_rate_stat_uncert" in result.data_vars
        assert "o_rate_stat_uncert" in result.data_vars

        # Check that intensities were calculated
        assert "h_intensity" in result.data_vars
        assert "o_intensity" in result.data_vars
        assert "h_intensity_stat_uncert" in result.data_vars
        assert "o_intensity_stat_uncert" in result.data_vars
        assert "h_intensity_sys_err" in result.data_vars
        assert "o_intensity_sys_err" in result.data_vars

        # Check that background intensities were calculated
        assert "h_bg_intensity" in result.data_vars
        assert "o_bg_intensity" in result.data_vars
        assert "h_bg_intensity_stat_uncert" in result.data_vars
        assert "o_bg_intensity_stat_uncert" in result.data_vars
        assert "h_bg_intensity_sys_err" in result.data_vars
        assert "o_bg_intensity_sys_err" in result.data_vars

        # Check that intermediate variables were cleaned up
        assert "h_counts_over_eff" not in result.data_vars
        assert "h_gf" not in result.data_vars
        assert "h_bg_rate_exposure_time" not in result.data_vars
        assert "o_bg_rate_exposure_time" not in result.data_vars


@pytest.mark.external_kernel
class TestIntegrationWithMocks:
    """Integration tests using mocked external dependencies."""

    def test_lo_l2_integration_minimal(
        self, minimal_pset, sample_geometric_factor_data
    ):
        """Test the main lo_l2 function with minimal mocking."""
        # This is a complex integration test - let's simplify it to just test
        # that the main function doesn't crash with proper mocking

        # Prepare input
        sci_dependencies = {"imap_lo_l1c_pset": [minimal_pset]}
        anc_dependencies = []
        descriptor = "l090-ena-h-sf-nsp-ram-hae-6deg-3mo"

        # Mock the complex external dependencies to return simple results
        with (
            patch(
                "imap_processing.lo.l2.lo_l2.create_sky_map_from_psets"
            ) as mock_create_map,
            patch(
                "imap_processing.lo.l2.lo_l2.load_geometric_factor_data"
            ) as mock_load_gf,
            patch(
                "imap_processing.lo.l2.lo_l2.calculate_all_rates_and_intensities"
            ) as mock_calc_rates,
        ):
            # Setup mocks to return minimal valid datasets
            h_gf_data, o_gf_data = sample_geometric_factor_data
            mock_load_gf.return_value = (h_gf_data, o_gf_data)

            # Mock the sky map creation to return a complete dataset
            mock_sky_map = Mock()
            mock_result_dataset = xr.Dataset(
                {
                    "h_intensity": (("epoch", "energy"), np.ones((1, 7))),
                    "o_intensity": (("epoch", "energy"), np.ones((1, 7)) * 0.5),
                    "exposure_time": (("epoch", "energy"), np.ones((1, 7))),
                }
            )
            mock_sky_map.to_dataset.return_value = mock_result_dataset
            mock_create_map.return_value = mock_sky_map

            # Mock the rates calculation to return the dataset unchanged
            mock_calc_rates.side_effect = lambda x: x

            # Run the function - should not crash
            result = lo_l2(sci_dependencies, anc_dependencies, descriptor)

            # Basic validation
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], xr.Dataset)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in various functions."""

    def test_lo_l2_no_pset_data(self):
        """Test error when no pointing set data is provided."""
        sci_dependencies = {}  # Missing imap_lo_l1c_pset
        anc_dependencies = []
        descriptor = "l090-ena-h-sf-nsp-ram-hae-6deg-3mo"

        with pytest.raises(ValueError, match="No pointing set data found"):
            lo_l2(sci_dependencies, anc_dependencies, descriptor)

    def test_create_sky_map_healpix_not_supported(self, minimal_pset):
        """Test error when HEALPix map is requested."""
        descriptor = "l090-ena-h-sf-nsp-ram-hnu-nside2-3mo"  # HEALPix descriptor

        with patch.object(MapDescriptor, "from_string") as mock_from_string:
            mock_map_desc = Mock()
            mock_healpix_map = Mock()  # Not a RectangularSkyMap
            mock_map_desc.to_empty_map.return_value = mock_healpix_map
            mock_from_string.return_value = mock_map_desc

            with pytest.raises(
                NotImplementedError, match="HEALPix map output not supported"
            ):
                create_sky_map_from_psets([minimal_pset], descriptor, pd.DataFrame())


# =============================================================================
# PROPERTY-BASED AND EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_efficiency_data_handling(self, minimal_pset):
        """Test handling of empty efficiency data."""
        empty_df = pd.DataFrame()
        result = add_efficiency_factors_to_pset(minimal_pset, empty_df)

        # Should create unity efficiency
        assert "efficiency" in result.data_vars
        np.testing.assert_array_equal(result["efficiency"].values, np.ones(7))

    def test_zero_exposure_time_handling(self):
        """Test handling of zero exposure times."""
        dataset = xr.Dataset(
            {
                "h_counts": (("energy",), np.ones(7) * 10),
                "o_counts": (("energy",), np.ones(7) * 5),
                "doubles_counts": (("energy",), np.ones(7) * 2),
                "triples_counts": (("energy",), np.ones(7) * 1),
                "exposure_time": (("energy",), np.zeros(7)),  # Zero exposure
            }
        )

        result = calculate_rates(dataset)

        # Should handle division by zero gracefully
        assert "h_rate" in result.data_vars
        # Rates should be infinite where exposure time is zero
        assert np.all(np.isinf(result["h_rate"].values))

    def test_negative_counts_handling(self):
        """Test handling of negative count values."""
        dataset = xr.Dataset(
            {
                "h_counts": (("energy",), np.array([-1, 0, 1, 2, 3, 4, 5])),
                "o_counts": (("energy",), np.array([0, 1, 2, 3, 4, 5, 6])),
                "doubles_counts": (("energy",), np.array([0, 0, 1, 1, 2, 2, 3])),
                "triples_counts": (("energy",), np.array([0, 0, 0, 1, 1, 1, 2])),
                "exposure_time": (("energy",), np.ones(7)),
            }
        )

        result = calculate_rates(dataset)

        # Should calculate rates even with negative counts
        assert "h_rate" in result.data_vars
        assert "h_rate_stat_uncert" in result.data_vars

        # Uncertainty calculation should handle negative counts
        # (sqrt of negative gives NaN, which is expected behavior)
        assert np.isnan(result["h_rate_stat_uncert"].values[0])
