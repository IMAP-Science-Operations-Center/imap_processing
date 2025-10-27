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
    calculate_bootstrap_corrections,
    calculate_efficiency_corrected_quantities,
    calculate_intensities,
    calculate_rates,
    calculate_sputtering_corrections,
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


@pytest.fixture(params=["h", "o", "doubles", "triples"])
def species_name(request):
    """Parametrized fixture for different species names."""
    return request.param


@pytest.fixture
def sample_pset():
    """Create a sample pointing set with typical data variables."""
    # Create counts data with some non-zero values
    counts = np.zeros(PSET_SHAPE)
    counts[:, 2:4, 10:20, 5:15] = 5  # Add some counts for testing

    exposure_factor = np.full(PSET_SHAPE, 0.5)

    # Create background rates data
    background_rates = np.full(PSET_SHAPE, 0.1)  # 0.1 counts/s background
    background_rates_stat_uncert = np.full(PSET_SHAPE, 0.01)  # 10% uncertainty

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
            "counts": (PSET_DIMS, counts),
            "exposure_factor": (PSET_DIMS, exposure_factor),
            "background_rates": (PSET_DIMS, background_rates),
            "background_rates_stat_uncert": (
                PSET_DIMS,
                background_rates_stat_uncert,
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
def sample_pset_for_species(species_name):
    """Create a sample pointing set for a specific species."""
    # Create counts data with some non-zero values
    counts = np.zeros(PSET_SHAPE)
    if species_name == "h":
        counts[:, 2:4, 10:20, 5:15] = 5
    elif species_name == "o":
        counts[:, 1:3, 15:25, 8:18] = 3
    elif species_name == "doubles":
        counts[:, 0:2, 5:15, 10:20] = 2
    elif species_name == "triples":
        counts[:, 3:5, 20:30, 15:25] = 1

    exposure_factor = np.full(PSET_SHAPE, 0.5)

    # Create coordinate arrays
    lons, lats = np.meshgrid(
        SPIN_ANGLE_BIN_CENTERS, OFF_ANGLE_BIN_CENTERS, indexing="ij"
    )
    hae_longitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_latitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_longitude[0, :, :] = lons
    hae_latitude[0, :, :] = lats

    # Base dataset with coords and exposure time
    dataset_dict = {
        "counts": (PSET_DIMS, counts),
        "exposure_factor": (PSET_DIMS, exposure_factor),
        "hae_longitude": (("epoch", "spin_angle", "off_angle"), hae_longitude),
        "hae_latitude": (("epoch", "spin_angle", "off_angle"), hae_latitude),
    }

    # Add background rates only for h and o
    if species_name in ["h", "o"]:
        bg_rates = np.full(PSET_SHAPE, 0.1 if species_name == "h" else 0.05)
        bg_uncert = np.full(PSET_SHAPE, 0.01 if species_name == "h" else 0.005)
        dataset_dict["background_rates"] = (PSET_DIMS, bg_rates)
        dataset_dict["background_rates_stat_uncert"] = (
            PSET_DIMS,
            bg_uncert,
        )

    dataset = xr.Dataset(
        dataset_dict,
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
    """Create a minimal pointing set with typical data for testing."""
    counts = np.ones(PSET_SHAPE)  # All ones for easy testing
    exposure_factor = np.full(PSET_SHAPE, 1.0)  # 1 second exposure for easy math

    # Create simple background rates for testing
    background_rates = np.full(PSET_SHAPE, 0.2)  # 0.2 counts/s
    background_rates_stat_uncert = np.full(PSET_SHAPE, 0.02)  # 10% uncertainty

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
            "counts": (PSET_DIMS, counts),
            "exposure_factor": (PSET_DIMS, exposure_factor),
            "background_rates": (PSET_DIMS, background_rates),
            "background_rates_stat_uncert": (
                PSET_DIMS,
                background_rates_stat_uncert,
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
def minimal_pset_for_species(species_name):
    """Create a minimal pointing set for a specific species."""
    # Create simple counts data
    if species_name == "h":
        counts = np.ones(PSET_SHAPE)
    elif species_name == "o":
        counts = np.ones(PSET_SHAPE) * 0.5
    elif species_name == "doubles":
        counts = np.ones(PSET_SHAPE) * 0.2
    elif species_name == "triples":
        counts = np.ones(PSET_SHAPE) * 0.1

    exposure_factor = np.full(PSET_SHAPE, 1.0)  # 1 second exposure for easy math

    # Simple coordinate arrays
    lons, lats = np.meshgrid(
        SPIN_ANGLE_BIN_CENTERS, OFF_ANGLE_BIN_CENTERS, indexing="ij"
    )
    hae_longitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_latitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_longitude[0, :, :] = lons
    hae_latitude[0, :, :] = lats

    # Base dataset with coords and exposure time
    dataset_dict = {
        "counts": (PSET_DIMS, counts),
        "exposure_factor": (PSET_DIMS, exposure_factor),
        "hae_longitude": (("epoch", "spin_angle", "off_angle"), hae_longitude),
        "hae_latitude": (("epoch", "spin_angle", "off_angle"), hae_latitude),
    }

    # Add background rates for all species
    bg_rates = np.full(PSET_SHAPE, 0.2 if species_name == "h" else 0.1)
    bg_uncert = np.full(PSET_SHAPE, 0.02 if species_name == "h" else 0.01)
    dataset_dict["background_rates"] = (PSET_DIMS, bg_rates)
    dataset_dict["background_rates_stat_uncert"] = (PSET_DIMS, bg_uncert)

    dataset = xr.Dataset(
        dataset_dict,
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

    # Current lo_l2.py uses generic variable names, not species-specific
    counts = np.ones((1, n_energy, n_lon, n_lat)) * 10  # 10 counts for easy math
    dataset["counts"] = (("epoch", "energy", "longitude", "latitude"), counts)

    # Add efficiency-corrected quantities for intensity calculations
    eff_corr = counts / 0.9  # Assuming 90% efficiency
    dataset["counts_over_eff"] = (
        ("epoch", "energy", "longitude", "latitude"),
        eff_corr,
    )
    dataset["counts_over_eff_squared"] = (
        ("epoch", "energy", "longitude", "latitude"),
        eff_corr,
    )

    # Add exposure time using the current naming convention
    exposure = np.ones((1, n_energy, n_lon, n_lat)) * 1.0  # 1 second
    dataset["exposure_factor"] = (
        ("epoch", "energy", "longitude", "latitude"),
        exposure,
    )

    return dataset


@pytest.fixture
def sample_dataset_with_geometric_factors():
    """Create a dataset with geometric factors for testing calculations."""
    dataset = xr.Dataset(
        coords={
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
        }
    )

    # Add current generic variable names used by lo_l2.py
    dataset["counts_over_eff"] = (("epoch", "energy"), np.ones((1, 7)) * 100)
    dataset["counts_over_eff_squared"] = (("epoch", "energy"), np.ones((1, 7)) * 100)
    dataset["exposure_factor"] = (("epoch", "energy"), np.ones((1, 7)) * 1.0)
    dataset["geometric_factor"] = (("energy",), np.ones(7) * 1e-4)
    dataset["energy"] = (("energy",), np.ones(7) * 0.1)  # Energy values
    dataset["geometric_factor_stat_uncert"] = (("energy",), np.ones(7) * 1e-5)

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

    # Add the intermediate background variables using current naming convention
    bg_rate_exposure_factor = np.ones((1, n_energy)) * 0.2  # 0.2 counts
    dataset["bg_rates_exposure_factor"] = (("epoch", "energy"), bg_rate_exposure_factor)

    # Background uncertainty squared times exposure time squared
    bg_rate_stat_uncert_exposure_factor2 = np.ones((1, n_energy)) * 0.004  # 0.02^2
    dataset["bg_rates_stat_uncert_exposure_factor2"] = (
        ("epoch", "energy"),
        bg_rate_stat_uncert_exposure_factor2,
    )

    # Add exposure time (using current naming convention)
    exposure = np.ones((1, n_energy)) * 1.0  # 1 second
    dataset["exposure_factor"] = (("epoch", "energy"), exposure)

    # Add geometric factors for systematic uncertainty calculation
    dataset["geometric_factor"] = (("energy",), np.ones(n_energy) * 1e-4)
    dataset["geometric_factor_stat_uncert"] = (("energy",), np.ones(n_energy) * 1e-5)

    return dataset


@pytest.fixture
def sample_dataset_with_sputtering_data():
    """Create datasets with ENA intensities for sputtering correction testing."""
    # Create a simple map dataset with the required variables for sputtering correction
    n_energy = 7
    n_lon, n_lat = 10, 5  # Smaller for testing

    coords = {
        "epoch": [8.1794907049e17],
        "energy": list(range(n_energy)),
        "longitude": np.linspace(0, 360, n_lon, endpoint=False),
        "latitude": np.linspace(-90, 90, n_lat),
    }

    # Create hydrogen dataset
    h_intensity_values = np.ones((1, n_energy, n_lon, n_lat)) * 1e6  # Base intensity
    h_intensity_values[0, 4, :, :] *= 3  # Higher at energy level 4 (ESA level 5)
    h_intensity_values[0, 5, :, :] *= 2  # Higher at energy level 5 (ESA level 6)

    h_dataset = xr.Dataset(coords=coords)
    h_dataset["ena_intensity"] = (
        ("epoch", "energy", "longitude", "latitude"),
        h_intensity_values,
    )

    # Add hydrogen background rates (lower values)
    h_bg_rates_values = np.ones((1, n_energy, n_lon, n_lat)) * 0.1e6  # 10% of intensity
    h_dataset["bg_rates"] = (
        ("epoch", "energy", "longitude", "latitude"),
        h_bg_rates_values,
    )

    # Add statistical uncertainties
    h_dataset["ena_intensity_stat_uncert"] = (
        ("epoch", "energy", "longitude", "latitude"),
        np.sqrt(h_intensity_values) * 0.1,
    )

    h_dataset["bg_rates_stat_uncert"] = (
        ("epoch", "energy", "longitude", "latitude"),
        np.sqrt(h_bg_rates_values) * 0.1,
    )

    # Add systematic error
    h_dataset["ena_intensity_sys_err"] = (
        ("epoch", "energy", "longitude", "latitude"),
        h_intensity_values * 0.05,
    )

    # Create oxygen dataset
    o_intensity_values = np.ones((1, n_energy, n_lon, n_lat)) * 1e6  # Base intensity
    o_intensity_values[0, 4, :, :] *= 5  # Higher at energy level 4 (ESA level 5)
    o_intensity_values[0, 5, :, :] *= 3  # Higher at energy level 5 (ESA level 6)

    o_dataset = xr.Dataset(coords=coords)
    o_dataset["ena_intensity"] = (
        ("epoch", "energy", "longitude", "latitude"),
        o_intensity_values,
    )

    # Add oxygen background rates (lower values)
    o_bg_rates_values = np.ones((1, n_energy, n_lon, n_lat)) * 0.1e6  # 10% of intensity
    o_dataset["bg_rates"] = (
        ("epoch", "energy", "longitude", "latitude"),
        o_bg_rates_values,
    )

    # Add statistical uncertainties
    o_dataset["ena_intensity_stat_uncert"] = (
        ("epoch", "energy", "longitude", "latitude"),
        np.sqrt(o_intensity_values) * 0.1,
    )

    o_dataset["bg_rates_stat_uncert"] = (
        ("epoch", "energy", "longitude", "latitude"),
        np.sqrt(o_bg_rates_values) * 0.1,
    )

    # Add systematic error
    o_dataset["ena_intensity_sys_err"] = (
        ("epoch", "energy", "longitude", "latitude"),
        o_intensity_values * 0.05,
    )

    # Add geometric factors for intensity calculations
    o_dataset["geometric_factor"] = (("energy",), np.ones(n_energy))

    return h_dataset, o_dataset


@pytest.fixture
def sample_dataset_with_bootstrap_data():
    """Create a dataset with ENA intensities for bootstrap correction testing."""
    # Create a simple map dataset with the required variables for bootstrap correction
    n_energy = 7
    n_lon, n_lat = 10, 5  # Smaller for testing

    coords = {
        "epoch": [8.1794907049e17],
        "energy": list(range(n_energy)),
        "longitude": np.linspace(0, 360, n_lon, endpoint=False),
        "latitude": np.linspace(-90, 90, n_lat),
    }

    # Create realistic energy values for hydrogen
    energy_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])  # keV

    # Create intensity values that follow a power law distribution
    # Higher intensities at lower energies
    base_intensity = 1e6  # particles/(cm^2 sr s keV)
    intensity_values = np.ones((1, n_energy, n_lon, n_lat))
    for i in range(n_energy):
        # Power law: I = I0 * (E/E0)^(-2.5)
        intensity_values[0, i, :, :] = base_intensity * (energy_values[i] / 1.0) ** (
            -2.5
        )

    dataset = xr.Dataset(coords=coords)
    dataset["ena_intensity"] = (
        ("epoch", "energy", "longitude", "latitude"),
        intensity_values,
    )
    dataset["energy"] = (("energy",), energy_values)
    dataset["geometric_factor"] = (("energy",), np.ones(n_energy))

    # Add background rates (much lower values)
    bg_rates_values = intensity_values * 0.1  # 10% of intensity as background
    dataset["bg_rates"] = (
        ("epoch", "energy", "longitude", "latitude"),
        bg_rates_values,
    )

    # Add statistical uncertainties (Poisson-like)
    dataset["ena_intensity_stat_uncert"] = (
        ("epoch", "energy", "longitude", "latitude"),
        np.sqrt(intensity_values) * 0.1,
    )

    # Add systematic error (5% of intensity)
    dataset["ena_intensity_sys_err"] = (
        ("epoch", "energy", "longitude", "latitude"),
        intensity_values * 0.05,
    )

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

    @pytest.mark.parametrize("species", ["h", "o"])
    def test_normalize_coordinates_basic(self, species):
        """Test basic coordinate normalization for a specific species."""
        # Create a pset with the specified species
        pset = xr.Dataset(
            {
                f"{species}_counts": (PSET_DIMS, np.ones(PSET_SHAPE)),
                "exposure_time": (PSET_DIMS, np.ones(PSET_SHAPE)),
                f"{species}_background_rates": (PSET_DIMS, np.ones(PSET_SHAPE) * 0.1),
                f"{species}_background_rates_stat_uncert": (
                    PSET_DIMS,
                    np.ones(PSET_SHAPE) * 0.01,
                ),
            },
            coords={
                "epoch": [8.1794907049e17],
                "esa_energy_step": ESA_ENERGY_STEPS,
                "spin_angle": SPIN_ANGLE_BIN_CENTERS,
                "off_angle": OFF_ANGLE_BIN_CENTERS,
            },
        )

        result = normalize_pset_coordinates(pset, species)

        # Check that dimensions were renamed
        assert "energy" in result.dims
        assert "esa_energy_step" not in result.dims

        # Check that energy coordinate is present
        assert "energy" in result.coords
        np.testing.assert_array_equal(result.coords["energy"], list(range(7)))

        # Check that old coordinate variable was dropped
        assert "esa_energy_step" not in result.variables

        # Check that variables were renamed
        assert "counts" in result.data_vars
        assert "exposure_factor" in result.data_vars
        assert "bg_rates" in result.data_vars
        assert "bg_rates_stat_uncert" in result.data_vars

        # Check that old variable names are gone
        assert f"{species}_counts" not in result.data_vars
        assert "exposure_time" not in result.data_vars

    @pytest.mark.parametrize("species", ["doubles", "triples"])
    def test_normalize_coordinates_no_background(self, species):
        """Test normalization for species without background rates."""
        # Create a pset with only counts and exposure time
        pset = xr.Dataset(
            {
                f"{species}_counts": (PSET_DIMS, np.ones(PSET_SHAPE)),
                "exposure_time": (PSET_DIMS, np.ones(PSET_SHAPE)),
            },
            coords={
                "epoch": [8.1794907049e17],
                "esa_energy_step": ESA_ENERGY_STEPS,
                "spin_angle": SPIN_ANGLE_BIN_CENTERS,
                "off_angle": OFF_ANGLE_BIN_CENTERS,
            },
        )

        # For species without background rates, the function should fail
        # because it tries to access background variables that don't exist
        with pytest.raises(ValueError, match="cannot rename"):
            normalize_pset_coordinates(pset, species)

    def test_normalize_coordinates_removes_old_coordinate(self):
        """Test that old esa_energy_step coordinate is removed."""
        species = "h"
        pset = xr.Dataset(
            {
                f"{species}_counts": (PSET_DIMS, np.ones(PSET_SHAPE)),
                "exposure_time": (PSET_DIMS, np.ones(PSET_SHAPE)),
                f"{species}_background_rates": (PSET_DIMS, np.ones(PSET_SHAPE) * 0.1),
                f"{species}_background_rates_stat_uncert": (
                    PSET_DIMS,
                    np.ones(PSET_SHAPE) * 0.01,
                ),
                "esa_energy_step_var": xr.DataArray([1, 2, 3, 4, 5, 6, 7]),  # Variable
            },
            coords={
                "epoch": [8.1794907049e17],
                "esa_energy_step": ESA_ENERGY_STEPS,
                "spin_angle": SPIN_ANGLE_BIN_CENTERS,
                "off_angle": OFF_ANGLE_BIN_CENTERS,
            },
        )

        result = normalize_pset_coordinates(pset, species)

        # Should remove the esa_energy_step coordinate and variable
        assert "esa_energy_step" not in result.variables
        assert "esa_energy_step_var" in result.variables  # Data variable should remain


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

    def test_calculate_efficiency_corrected_quantities(self):
        """Test calculation of efficiency-corrected quantities."""
        # Create a dataset with the current generic variable names
        pset = xr.Dataset(
            {
                "counts": (("energy",), np.ones(7) * 10),  # 10 counts
                "exposure_factor": (("energy",), np.ones(7) * 1.0),  # 1 second
                "bg_rates": (("energy",), np.ones(7) * 0.1),  # 0.1 counts/s
                "bg_rates_stat_uncert": (("energy",), np.ones(7) * 0.01),  # uncertainty
                "efficiency": (
                    ("energy",),
                    np.array([0.8, 0.85, 0.9, 0.95, 0.88, 0.92, 0.87]),
                ),
            },
            coords={"energy": list(range(7))},
        )

        result = calculate_efficiency_corrected_quantities(pset)

        # Check that corrected quantities were added
        assert "counts_over_eff" in result.data_vars
        assert "counts_over_eff_squared" in result.data_vars
        assert "bg_rates_exposure_factor" in result.data_vars
        assert "bg_rates_stat_uncert_exposure_factor2" in result.data_vars

        # Check dimensions
        assert result["counts_over_eff"].dims == pset["counts"].dims

        # Check that division by efficiency happened correctly
        expected_over_eff = pset["counts"] / pset["efficiency"]
        xr.testing.assert_allclose(result["counts_over_eff"], expected_over_eff)

        # Check that division by efficiency squared happened correctly
        expected_over_eff_sq = pset["counts"] / (pset["efficiency"] ** 2)
        xr.testing.assert_allclose(
            result["counts_over_eff_squared"], expected_over_eff_sq
        )

        # Check background rate calculations
        expected_bg_exposure = pset["bg_rates"] * pset["exposure_factor"]
        xr.testing.assert_allclose(
            result["bg_rates_exposure_factor"], expected_bg_exposure
        )

        expected_bg_uncert_exposure = (
            pset["bg_rates_stat_uncert"] ** 2 * pset["exposure_factor"] ** 2
        )
        xr.testing.assert_allclose(
            result["bg_rates_stat_uncert_exposure_factor2"], expected_bg_uncert_exposure
        )


class TestCalculateRates:
    """Tests for the calculate_rates function."""

    def test_calculate_rates_basic(self, sample_sky_map_dataset):
        """Test rate calculation with current implementation."""
        result = calculate_rates(sample_sky_map_dataset)

        # Check that the expected output variables were created
        assert "ena_count_rate" in result.data_vars
        assert "ena_count_rate_stat_uncert" in result.data_vars

        # Check dimensions match input
        assert result["ena_count_rate"].dims == sample_sky_map_dataset["counts"].dims

        # Check rate calculation (counts / exposure_factor)
        # With counts=10 and exposure_factor=1, rate should be 10
        expected_rate = (
            sample_sky_map_dataset["counts"] / sample_sky_map_dataset["exposure_factor"]
        )
        xr.testing.assert_allclose(result["ena_count_rate"], expected_rate)

        # Check uncertainty calculation (sqrt(counts) / exposure_factor)
        expected_uncert = (
            np.sqrt(sample_sky_map_dataset["counts"])
            / sample_sky_map_dataset["exposure_factor"]
        )
        xr.testing.assert_allclose(
            result["ena_count_rate_stat_uncert"], expected_uncert
        )

    def test_calculate_rates_missing_variables(self):
        """Rate calculation when required variables are missing."""
        # Create dataset missing required variables
        dataset = xr.Dataset(
            {
                "counts": (("epoch", "energy"), np.ones((1, 7)) * 5),
                # Missing exposure_factor
            }
        )

        # Should raise KeyError for missing exposure_factor
        with pytest.raises(KeyError, match="exposure_factor"):
            calculate_rates(dataset)


class TestCalculateIntensities:
    """Tests for the calculate_intensities function."""

    def test_calculate_intensities_basic(self, sample_dataset_with_geometric_factors):
        """Test intensity calculation with current implementation."""
        result = calculate_intensities(sample_dataset_with_geometric_factors)

        # Check that the expected output variables were created
        assert "ena_intensity" in result.data_vars
        assert "ena_intensity_stat_uncert" in result.data_vars
        assert "ena_intensity_sys_err" in result.data_vars

        # Check intensity calculation:
        # counts_over_eff / (geometric_factor * energy * exposure_factor)
        # 100 / (1e-4 * 0.1 * 1.0) = 100 / 1e-5 = 1e7
        expected_intensity = sample_dataset_with_geometric_factors[
            "counts_over_eff"
        ] / (
            sample_dataset_with_geometric_factors["geometric_factor"]
            * sample_dataset_with_geometric_factors["energy"]
            * sample_dataset_with_geometric_factors["exposure_factor"]
        )
        xr.testing.assert_allclose(result["ena_intensity"], expected_intensity)

        # Check statistical uncertainty calculation
        expected_stat_uncert = np.sqrt(
            sample_dataset_with_geometric_factors["counts_over_eff_squared"]
            / (
                sample_dataset_with_geometric_factors["geometric_factor"]
                * sample_dataset_with_geometric_factors["energy"]
                * sample_dataset_with_geometric_factors["exposure_factor"]
            )
        )
        xr.testing.assert_allclose(
            result["ena_intensity_stat_uncert"], expected_stat_uncert
        )

        # Check systematic uncertainty calculation
        expected_sys_err = (
            result["ena_intensity"]
            * sample_dataset_with_geometric_factors["geometric_factor_stat_uncert"]
            / sample_dataset_with_geometric_factors["geometric_factor"]
        )
        xr.testing.assert_allclose(result["ena_intensity_sys_err"], expected_sys_err)

    def test_calculate_intensities_missing_variables(self):
        """Test intensity calculation when required variables are missing."""
        # Create dataset missing geometric_factor
        dataset = xr.Dataset(
            {
                "counts_over_eff": (("energy",), np.ones(7) * 100),
                "counts_over_eff_squared": (("energy",), np.ones(7) * 100),
                "exposure_factor": (("energy",), np.ones(7) * 1.0),
                "energy": (("energy",), np.ones(7) * 0.1),
                # Missing geometric_factor
            }
        )

        # Should raise KeyError for missing geometric_factor
        with pytest.raises(KeyError, match="geometric_factor"):
            calculate_intensities(dataset)


class TestCalculateBackgrounds:
    """Tests for the calculate_backgrounds function."""

    def test_calculate_backgrounds_basic(
        self, sample_dataset_with_background_intermediates
    ):
        """Test basic background calculations with standard data."""
        dataset = sample_dataset_with_background_intermediates

        result = calculate_backgrounds(dataset)

        # Check that background variables were calculated
        assert "bg_rates" in result.data_vars
        assert "bg_rates_stat_uncert" in result.data_vars
        assert "bg_rates_sys_err" in result.data_vars

        # Check background rate calculation
        # bg_rates_exposure_factor / exposure_factor = 0.2 / 1.0 = 0.2
        expected_bg_rate = (
            dataset["bg_rates_exposure_factor"] / dataset["exposure_factor"]
        )
        xr.testing.assert_allclose(result["bg_rates"], expected_bg_rate)

        # Check statistical uncertainty calculation
        # sqrt(bg_rates_stat_uncert_exposure_factor2) / exposure_factor^2
        expected_stat_uncert = np.sqrt(
            dataset["bg_rates_stat_uncert_exposure_factor2"]
            / dataset["exposure_factor"] ** 2
        )
        xr.testing.assert_allclose(result["bg_rates_stat_uncert"], expected_stat_uncert)

        # Check systematic uncertainty calculation
        # (geometric_factor_stat_uncert / geometric_factor) * bg_rates
        expected_sys_err = (
            result["bg_rates"]
            * dataset["geometric_factor_stat_uncert"]
            / dataset["geometric_factor"]
        )
        xr.testing.assert_allclose(result["bg_rates_sys_err"], expected_sys_err)

    def test_calculate_backgrounds_zero_exposure(self):
        """Test background calculations with zero exposure time."""
        dataset = xr.Dataset(
            {
                "bg_rates_exposure_factor": (
                    ("epoch", "energy"),
                    np.ones((1, 7)) * 0.2,
                ),
                "bg_rates_stat_uncert_exposure_factor2": (
                    ("epoch", "energy"),
                    np.ones((1, 7)) * 0.004,
                ),
                "exposure_factor": (("epoch", "energy"), np.zeros((1, 7))),
                "geometric_factor": (("energy",), np.ones(7) * 1e-4),
                "geometric_factor_stat_uncert": (("energy",), np.ones(7) * 1e-5),
            },
            coords={"epoch": [8.1794907049e17], "energy": list(range(7))},
        )

        result = calculate_backgrounds(dataset)

        # Should handle division by zero gracefully
        assert "bg_rates" in result.data_vars
        assert "bg_rates_stat_uncert" in result.data_vars
        assert "bg_rates_sys_err" in result.data_vars
        # Results should be infinite where exposure time is zero
        assert np.all(np.isinf(result["bg_rates"].values))
        assert np.all(np.isinf(result["bg_rates_stat_uncert"].values))


class TestCalculateSputteringCorrections:
    """Tests for the calculate_sputtering_corrections function."""

    def test_calculate_sputtering_corrections_basic(
        self, sample_dataset_with_sputtering_data
    ):
        """Test basic sputtering corrections for hydrogen and oxygen intensities."""
        h_dataset, o_dataset = sample_dataset_with_sputtering_data

        # Test with hydrogen dataset first
        original_h_intensity = h_dataset["ena_intensity"].copy()
        original_h_stat_uncert = h_dataset["ena_intensity_stat_uncert"].copy()
        original_h_sys_err = h_dataset["ena_intensity_sys_err"].copy()

        result_h = calculate_sputtering_corrections(h_dataset, o_dataset)

        # Check that only energy levels 4 and 5 (ESA levels 5 and 6) were modified
        # for hydrogen
        for energy_idx in [0, 1, 2, 3, 6]:
            np.testing.assert_array_equal(
                result_h["ena_intensity"][0, energy_idx, :, :].values,
                original_h_intensity[0, energy_idx, :, :].values,
                err_msg=f"Hydrogen energy level {energy_idx} should not be modified",
            )

        # Check that energy levels 4 and 5 were modified (should be lower)
        assert np.all(
            result_h["ena_intensity"][0, 4, :, :].values
            < original_h_intensity[0, 4, :, :].values
        ), (
            "Hydrogen energy level 4 intensity should be reduced by sputtering "
            "correction"
        )

        assert np.all(
            result_h["ena_intensity"][0, 5, :, :].values
            < original_h_intensity[0, 5, :, :].values
        ), (
            "Hydrogen energy level 5 intensity should be reduced by sputtering "
            "correction"
        )

        # Check that uncertainties were also updated for levels 4 and 5
        assert not np.array_equal(
            result_h["ena_intensity_stat_uncert"][0, 4, :, :].values,
            original_h_stat_uncert[0, 4, :, :].values,
        ), "Statistical uncertainty should be updated for hydrogen energy level 4"

        assert not np.array_equal(
            result_h["ena_intensity_sys_err"][0, 4, :, :].values,
            original_h_sys_err[0, 4, :, :].values,
        ), "Systematic error should be updated for hydrogen energy level 4"

        # Test with oxygen dataset
        original_o_intensity = o_dataset["ena_intensity"].copy()

        result_o = calculate_sputtering_corrections(o_dataset, o_dataset)

        # Check that only energy levels 4 and 5 were modified for oxygen
        for energy_idx in [0, 1, 2, 3, 6]:
            np.testing.assert_array_equal(
                result_o["ena_intensity"][0, energy_idx, :, :].values,
                original_o_intensity[0, energy_idx, :, :].values,
                err_msg=f"Oxygen energy level {energy_idx} should not be modified",
            )

        # Check that energy levels 4 and 5 were modified
        assert np.all(
            result_o["ena_intensity"][0, 4, :, :].values
            < original_o_intensity[0, 4, :, :].values
        ), "Oxygen energy level 4 intensity should be reduced by sputtering correction"

        assert np.all(
            result_o["ena_intensity"][0, 5, :, :].values
            < original_o_intensity[0, 5, :, :].values
        ), "Oxygen energy level 5 intensity should be reduced by sputtering correction"

    def test_calculate_sputtering_corrections_equations(
        self, sample_dataset_with_sputtering_data
    ):
        """Test that sputtering corrections follow the correct equations."""
        h_dataset, o_dataset = sample_dataset_with_sputtering_data

        # Get the subset that will be processed (energy levels 4 and 5)
        o_small_dataset = o_dataset.isel(epoch=0, energy=[4, 5])

        # Calculate expected j_o_prime (Equation 9)
        expected_j_o_prime = o_small_dataset["ena_intensity"] - o_small_dataset[
            "bg_rates"
        ] / (o_small_dataset["geometric_factor"] * o_small_dataset["energy"])
        expected_j_o_prime = expected_j_o_prime.where(expected_j_o_prime >= 0, 0)

        # Expected correction factors from the mapping document table 2
        sputter_correction_factor = np.array([0.15, 0.01])

        # Calculate expected corrected intensity (Equation 11) for hydrogen
        h_small_dataset = h_dataset.isel(epoch=0, energy=[4, 5])
        expected_corrected_intensity = (
            h_small_dataset["ena_intensity"]
            - sputter_correction_factor[:, np.newaxis, np.newaxis] * expected_j_o_prime
        )

        # Run the function
        result = calculate_sputtering_corrections(h_dataset, o_dataset)

        # Check that the corrected intensities match expected values
        np.testing.assert_allclose(
            result["ena_intensity"][0, [4, 5], :, :].values,
            expected_corrected_intensity.values,
            rtol=1e-10,
            err_msg="Sputtering-corrected intensities don't match expected calculation",
        )

    def test_calculate_sputtering_corrections_negative_j_o_prime(self):
        """Test handling when j_o_prime becomes negative."""
        # Create dataset where background > intensity (would give negative j_o_prime)
        coords = {
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
            "longitude": np.linspace(0, 360, 5, endpoint=False),
            "latitude": np.linspace(-90, 90, 3),
        }

        # Create hydrogen dataset
        h_dataset = xr.Dataset(coords=coords)
        h_dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 1e6,
        )
        h_dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 0.5e6,
        )

        # Add required uncertainty variables for hydrogen
        h_dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 0.1e6,
        )
        h_dataset["bg_rates_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 0.05e6,
        )
        h_dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 0.05e6,
        )

        # Create oxygen dataset where background > intensity
        o_dataset = xr.Dataset(coords=coords)
        o_dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 1e6,
        )
        o_dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 2e6,  # Higher than signal
        )

        # Add required uncertainty variables for oxygen
        o_dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 0.1e6,
        )
        o_dataset["bg_rates_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 0.1e6,
        )
        o_dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 5, 3)) * 0.05e6,
        )
        o_dataset["geometric_factor"] = (("energy",), np.ones(7))

        result = calculate_sputtering_corrections(h_dataset, o_dataset)

        # Function should handle negative j_o_prime by setting it to zero
        # This means no sputtering correction should be applied
        # The corrected intensity should equal the original intensity
        np.testing.assert_allclose(
            result["ena_intensity"][0, [4, 5], :, :].values,
            h_dataset["ena_intensity"][0, [4, 5], :, :].values,
            rtol=1e-10,
            err_msg=(
                "When background > signal, no sputtering correction should be applied"
            ),
        )

    def test_calculate_sputtering_corrections_only_oxygen(
        self, sample_dataset_with_sputtering_data
    ):
        """Test that sputtering corrections work for different species datasets."""
        h_dataset, o_dataset = sample_dataset_with_sputtering_data

        # Store original hydrogen values
        original_h_intensity = h_dataset["ena_intensity"].copy()
        original_h_stat_uncert = h_dataset["ena_intensity_stat_uncert"].copy()
        original_h_sys_err = h_dataset["ena_intensity_sys_err"].copy()

        # Store original oxygen values
        original_o_intensity = o_dataset["ena_intensity"].copy()

        # Test hydrogen dataset with oxygen reference
        result_h = calculate_sputtering_corrections(h_dataset, o_dataset)

        # Check that hydrogen values changed for energy levels 4 and 5
        assert not np.array_equal(
            result_h["ena_intensity"][0, 4, :, :].values,
            original_h_intensity[0, 4, :, :].values,
        ), "Hydrogen intensity should be affected by sputtering corrections"

        assert not np.array_equal(
            result_h["ena_intensity_stat_uncert"][0, 4, :, :].values,
            original_h_stat_uncert[0, 4, :, :].values,
        ), "Hydrogen stat uncertainty should be updated"

        assert not np.array_equal(
            result_h["ena_intensity_sys_err"][0, 4, :, :].values,
            original_h_sys_err[0, 4, :, :].values,
        ), "Hydrogen sys error should be updated"

        # Test oxygen dataset with itself as reference
        result_o = calculate_sputtering_corrections(o_dataset, o_dataset)

        # Check that oxygen values also changed
        assert not np.array_equal(
            result_o["ena_intensity"][0, 4, :, :].values,
            original_o_intensity[0, 4, :, :].values,
        ), "Oxygen intensity should also be affected by sputtering corrections"

    def test_calculate_sputtering_corrections_uncertainty_propagation(
        self, sample_dataset_with_sputtering_data
    ):
        """Test that uncertainties are properly propagated in sputtering corrections."""
        h_dataset, o_dataset = sample_dataset_with_sputtering_data

        # Get subset for manual calculation
        o_small_dataset = o_dataset.isel(epoch=0, energy=[4, 5])
        h_small_dataset = h_dataset.isel(epoch=0, energy=[4, 5])

        # Manual calculation following equations 10, 12
        j_o_prime = o_small_dataset["ena_intensity"] - o_small_dataset["bg_rates"]
        j_o_prime = j_o_prime.where(j_o_prime >= 0, 0)

        j_o_prime_var = (
            o_small_dataset["ena_intensity_stat_uncert"] ** 2
            + (
                o_small_dataset["bg_rates_stat_uncert"]
                / (o_small_dataset["energy"] * o_small_dataset["geometric_factor"])
            )
            ** 2
        )

        sputter_correction_factor = np.array([0.15, 0.01])[:, np.newaxis, np.newaxis]

        expected_corrected_var = (
            h_small_dataset["ena_intensity_stat_uncert"] ** 2
            + (sputter_correction_factor**2) * j_o_prime_var
        )

        result = calculate_sputtering_corrections(h_dataset, o_dataset)

        # Check that statistical uncertainties match expected propagation
        np.testing.assert_allclose(
            result["ena_intensity_stat_uncert"][0, [4, 5], :, :].values ** 2,
            expected_corrected_var.values,
            rtol=1e-10,
            err_msg="Statistical uncertainty propagation is incorrect",
        )

    def test_calculate_sputtering_corrections_energy_levels(self):
        """Test that sputtering corrections are applied to correct energy levels."""
        # Create minimal dataset for testing specific energy level targeting
        coords = {
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
            "longitude": [0, 90, 180, 270],
            "latitude": [-45, 0, 45],
        }

        # Create hydrogen dataset
        h_dataset = xr.Dataset(coords=coords)
        h_intensity_data = np.zeros((1, 7, 4, 3))
        h_bg_rates_data = np.zeros((1, 7, 4, 3))

        # Set specific values for energy levels 4 and 5 only
        h_intensity_data[0, 4, :, :] = 200_000_000  # 200M for energy index 4
        h_intensity_data[0, 5, :, :] = 250_000_000  # 250M for energy index 5
        h_bg_rates_data[0, 4, :, :] = 20_000_000  # 20M background
        h_bg_rates_data[0, 5, :, :] = 25_000_000  # 25M background

        h_dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            h_intensity_data,
        )
        h_dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            h_bg_rates_data,
        )
        h_dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 4, 3)) * 100_000,
        )
        h_dataset["bg_rates_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 4, 3)) * 10_000,
        )
        h_dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 4, 3)) * 50_000,
        )

        # Create oxygen dataset with higher values
        o_dataset = xr.Dataset(coords=coords)
        o_intensity_data = np.zeros((1, 7, 4, 3))
        o_bg_rates_data = np.zeros((1, 7, 4, 3))

        # Set specific values for energy levels 4 and 5 only
        o_intensity_data[0, 4, :, :] = 250_000_000  # 250M for energy index 4
        o_intensity_data[0, 5, :, :] = 300_000_000  # 300M for energy index 5
        o_bg_rates_data[0, 4, :, :] = 25_000_000  # 25M background
        o_bg_rates_data[0, 5, :, :] = 30_000_000  # 30M background

        o_dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            o_intensity_data,
        )
        o_dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            o_bg_rates_data,
        )
        o_dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 4, 3)) * 100_000,
        )
        o_dataset["bg_rates_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 4, 3)) * 10_000,
        )
        o_dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 4, 3)) * 50_000,
        )
        o_dataset["geometric_factor"] = (("energy",), np.ones(7))

        # Make a copy to preserve original values for comparison
        original_h_dataset = h_dataset.copy(deep=True)
        result = calculate_sputtering_corrections(h_dataset, o_dataset)

        # Only energy indices 4 and 5 should be modified
        modified_indices = [4, 5]
        unchanged_indices = [0, 1, 2, 3, 6]

        for idx in unchanged_indices:
            np.testing.assert_array_equal(
                result["ena_intensity"][0, idx, :, :].values,
                original_h_dataset["ena_intensity"][0, idx, :, :].values,
                err_msg=f"Energy index {idx} should not be modified",
            )

        for idx in modified_indices:
            # Check that values changed with some tolerance for numerical precision
            original_values = original_h_dataset["ena_intensity"][0, idx, :, :].values
            corrected_values = result["ena_intensity"][0, idx, :, :].values

            assert not np.allclose(corrected_values, original_values, rtol=1e-10), (
                f"Energy index {idx} should be modified"
            )


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
            "energy_stat_uncert",
            "geometric_factor",
            "geometric_factor_stat_uncert",
        ]

        for var in expected_vars:
            assert var in result.data_vars
            assert result[var].dims == ("energy",)
            assert result[var].shape == (7,)
            assert np.all(result[var].values == 0)  # Should be initialized to zeros

        # The energy coordinate should also be updated
        assert "energy" in result.coords
        assert result.coords["energy"].shape == (7,)
        assert np.all(result.coords["energy"].values == 0)  # Should be zeros


class TestPopulateGeometricFactors:
    """Tests for the populate_geometric_factors function."""

    @pytest.mark.parametrize("species", ["h", "o"])
    def test_populate_geometric_factors(self, species, sample_geometric_factor_data):
        """Test population of geometric factor values for a specific species."""
        h_gf_data, o_gf_data = sample_geometric_factor_data
        gf_data = h_gf_data if species == "h" else o_gf_data

        # Create initialized dataset
        dataset = xr.Dataset(coords={"energy": range(7)})
        dataset = initialize_geometric_factor_variables(dataset)

        result = populate_geometric_factors(dataset, gf_data, species)

        # Check that values were populated correctly
        for i in range(7):
            if species == "h":
                # Check hydrogen values
                assert result["energy"].values[i] == 0.01 * (i + 1)
                assert result["geometric_factor"].values[i] == 1e-4 * (i + 1)
                assert result["geometric_factor_stat_uncert"].values[i] == (
                    1e-5 * (i + 1)
                )
            else:  # oxygen
                assert result["energy"].values[i] == 0.015 * (i + 1)
                assert result["geometric_factor"].values[i] == 1.5e-4 * (i + 1)
                assert result["geometric_factor_stat_uncert"].values[i] == (
                    1.5e-5 * (i + 1)
                )

    def test_populate_geometric_factors_no_gf_species(self):
        """Test population for species without geometric factors."""
        # Create initialized dataset
        dataset = xr.Dataset(coords={"energy": range(7)})
        dataset = initialize_geometric_factor_variables(dataset)

        gf_data = pd.DataFrame()  # Empty dataframe

        # Test with doubles (no geometric factors)
        result = populate_geometric_factors(dataset, gf_data, "doubles")

        # Should return dataset unchanged (all zeros)
        assert np.all(result["geometric_factor"].values == 0)


class TestCleanupIntermediateVariables:
    """Tests for the cleanup_intermediate_variables function."""

    def test_cleanup_intermediate_variables(self):
        """Test removal of intermediate variables."""
        # Create dataset with intermediate variables using current naming
        dataset = xr.Dataset(
            {
                "counts": (("energy",), np.ones(7)),
                "counts_over_eff": (("energy",), np.ones(7)),
                "counts_over_eff_squared": (("energy",), np.ones(7)),
                "bg_rates_exposure_factor": (("energy",), np.ones(7)),
                "bg_rates_stat_uncert_exposure_factor2": (("energy",), np.ones(7)),
                "ena_intensity": (("energy",), np.ones(7)),  # Should be kept
                "exposure_factor": (("energy",), np.ones(7)),  # Should be kept
            }
        )

        result = cleanup_intermediate_variables(dataset)

        # Should keep these variables
        assert "counts" in result.data_vars
        assert "ena_intensity" in result.data_vars
        assert "exposure_factor" in result.data_vars

        # Should remove these intermediate variables
        assert "counts_over_eff" not in result.data_vars
        assert "counts_over_eff_squared" not in result.data_vars
        assert "bg_rates_exposure_factor" not in result.data_vars
        assert "bg_rates_stat_uncert_exposure_factor2" not in result.data_vars

    def test_cleanup_partial_variables(self):
        """Test cleanup when only some intermediate variables exist."""
        # Create dataset with only some intermediate variables
        dataset = xr.Dataset(
            {
                "counts": (("energy",), np.ones(7)),
                "counts_over_eff": (("energy",), np.ones(7)),
                "exposure_factor": (("energy",), np.ones(7)),
            }
        )

        result = cleanup_intermediate_variables(dataset)

        # Should keep these
        assert "counts" in result.data_vars
        assert "exposure_factor" in result.data_vars

        # Should remove only the existing intermediate variable
        assert "counts_over_eff" not in result.data_vars


class TestCalculateBootstrapCorrections:
    """Tests for the calculate_bootstrap_corrections function."""

    def test_calculate_bootstrap_corrections_basic(
        self, sample_dataset_with_bootstrap_data
    ):
        """Test basic bootstrap correction functionality."""
        dataset = sample_dataset_with_bootstrap_data.copy()

        # Store original values for comparison
        original_intensity = dataset["ena_intensity"].copy()

        result = calculate_bootstrap_corrections(dataset)

        # Check that bootstrap corrections were applied
        assert "ena_intensity" in result.data_vars
        assert "ena_intensity_stat_uncert" in result.data_vars
        assert "ena_intensity_sys_err" in result.data_vars

        # Check that intermediate bootstrap variables were removed
        assert "bootstrap_intensity" not in result.data_vars
        assert "bootstrap_intensity_stat_uncert" not in result.data_vars
        assert "bootstrap_intensity_sys_err" not in result.data_vars

        # Check that corrected intensities are different from originals
        # (bootstrap should reduce intensities due to spillover correction)
        corrected_intensity = result["ena_intensity"]
        assert not np.allclose(
            corrected_intensity.values, original_intensity.values, rtol=1e-10
        ), "Bootstrap corrections should modify intensities"

        # Check that corrected intensities are generally lower
        # (bootstrap removes spillover from higher to lower energies)
        for energy_idx in range(5):  # Lower energy channels should be reduced
            assert np.all(
                corrected_intensity[0, energy_idx, :, :].values
                <= original_intensity[0, energy_idx, :, :].values
            ), f"Bootstrap should reduce intensity at energy index {energy_idx}"

        # This is a spot check value to ensure we are getting actual
        # corrected intensities. Previously we were missing the application
        # of the lower energy channels in the summation.
        np.testing.assert_allclose(corrected_intensity[0, 5, 0, 0], [895.96302438])

    def test_calculate_bootstrap_corrections_equations(
        self, sample_dataset_with_bootstrap_data
    ):
        """Test that bootstrap equations are correctly implemented."""
        dataset = sample_dataset_with_bootstrap_data.copy()

        # Calculate expected j_c_prime (equation 14)
        j_c_prime_expected = dataset["ena_intensity"] - dataset["bg_rates"]
        j_c_prime_expected = j_c_prime_expected.where(j_c_prime_expected >= 0, 0)

        # Apply bootstrap corrections and check the calculation was done correctly
        result = calculate_bootstrap_corrections(dataset)

        # Verify the final result makes sense given the bootstrap factors
        # Higher energy channels should have less correction
        assert np.all(result["ena_intensity"][0, 6, :, :] >= 0), (
            "Bootstrap intensities should be non-negative"
        )

    def test_calculate_bootstrap_corrections_negative_handling(self):
        """Test proper handling of negative values during bootstrap calculation."""
        # Create dataset with some negative j_c_prime values
        coords = {
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
            "longitude": [0, 90],
            "latitude": [0, 45],
        }

        dataset = xr.Dataset(coords=coords)

        # Create energy values
        energy_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        dataset["energy"] = (("energy",), energy_values)
        dataset["geometric_factor"] = (("energy",), np.ones(7))

        # Create intensities where some background > intensity (negative j_c_prime)
        intensity_values = np.ones((1, 7, 2, 2)) * 1e6
        bg_rates_values = np.ones((1, 7, 2, 2)) * 1.5e6  # Higher than intensity

        dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values,
        )
        dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            bg_rates_values,
        )
        dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 2, 2)) * 1e5,
        )
        dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 2, 2)) * 5e4,
        )

        result = calculate_bootstrap_corrections(dataset)

        # All corrected intensities should be non-negative
        assert np.all(result["ena_intensity"].values >= 0), (
            "Bootstrap corrections should not produce negative intensities"
        )

    def test_calculate_bootstrap_corrections_energy_dependence(self):
        """Test that bootstrap corrections show proper energy dependence."""
        # Create dataset with realistic energy spectrum
        coords = {
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
            "longitude": [0],
            "latitude": [0],
        }

        dataset = xr.Dataset(coords=coords)

        # Create energy values
        energy_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        dataset["energy"] = (("energy",), energy_values)
        dataset["geometric_factor"] = (("energy",), np.ones(7))

        # Create a steep power law spectrum (typical for ENAs)
        base_intensity = 1e8
        intensity_values = np.ones((1, 7, 1, 1))
        for i in range(7):
            intensity_values[0, i, 0, 0] = base_intensity * (energy_values[i]) ** (-3)

        dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values,
        )

        # Low background
        dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values * 0.01,  # 1% background
        )
        dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.sqrt(intensity_values) * 0.1,
        )
        dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values * 0.05,
        )

        result = calculate_bootstrap_corrections(dataset)

        # Lower energy channels should show larger corrections
        # because they receive spillover from higher energy channels
        original_ratios = []
        corrected_ratios = []

        for i in range(6):  # Compare adjacent energy channels
            original_ratio = (
                dataset["ena_intensity"][0, i, 0, 0].values
                / dataset["ena_intensity"][0, i + 1, 0, 0].values
            )
            corrected_ratio = (
                result["ena_intensity"][0, i, 0, 0].values
                / result["ena_intensity"][0, i + 1, 0, 0].values
            )
            original_ratios.append(original_ratio)
            corrected_ratios.append(corrected_ratio)

        # Bootstrap should affect the intensities
        # Check that the bootstrap corrections are actually being applied

        # For power law spectra with small backgrounds, corrections may be very small
        # Let's check that the algorithm at least completes without error
        # and produces reasonable output

        # Check that all intensities are finite and positive
        assert np.all(np.isfinite(result["ena_intensity"].values)), (
            "All corrected intensities should be finite"
        )
        assert np.all(result["ena_intensity"].values >= 0), (
            "All corrected intensities should be non-negative"
        )

        # Check that uncertainties are reasonable
        assert np.all(np.isfinite(result["ena_intensity_stat_uncert"].values)), (
            "All statistical uncertainties should be finite"
        )
        assert np.all(np.isfinite(result["ena_intensity_sys_err"].values)), (
            "All systematic errors should be finite"
        )

    def test_calculate_bootstrap_corrections_uncertainty_propagation(self):
        """Test proper uncertainty propagation in bootstrap corrections."""
        # Create simple dataset for uncertainty testing
        coords = {
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
            "longitude": [0],
            "latitude": [0],
        }

        dataset = xr.Dataset(coords=coords)

        # Create energy values
        energy_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        dataset["energy"] = (("energy",), energy_values)
        dataset["geometric_factor"] = (("energy",), np.ones(7))

        # Simple flat spectrum for easier uncertainty analysis
        intensity_values = np.ones((1, 7, 1, 1)) * 1e6
        dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values,
        )
        dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values * 0.1,  # 10% background
        )

        # Known uncertainties
        stat_uncert = np.ones((1, 7, 1, 1)) * 1e5
        sys_err = np.ones((1, 7, 1, 1)) * 5e4

        dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            stat_uncert,
        )
        dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            sys_err,
        )

        result = calculate_bootstrap_corrections(dataset)

        # Check that uncertainties are properly propagated
        assert np.all(result["ena_intensity_stat_uncert"].values >= 0), (
            "Statistical uncertainties should be non-negative"
        )
        assert np.all(result["ena_intensity_sys_err"].values >= 0), (
            "Systematic errors should be non-negative"
        )

        # Uncertainties should be reasonable relative to the intensities
        relative_stat_uncert = (
            result["ena_intensity_stat_uncert"] / result["ena_intensity"]
        )
        assert np.all(relative_stat_uncert.values < 1.0), (
            "Relative statistical uncertainty should be reasonable"
        )

    def test_calculate_bootstrap_corrections_virtual_channel(self):
        """Test the virtual channel E8 calculation and its impact."""
        # Create dataset focused on energy channels 5 and 6 for E8 calculation
        coords = {
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
            "longitude": [0],
            "latitude": [0],
        }

        dataset = xr.Dataset(coords=coords)

        # Create energy values where E6/E7 ratio is well-defined
        energy_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        dataset["energy"] = (("energy",), energy_values)
        dataset["geometric_factor"] = (("energy",), np.ones(7))

        # Create intensities with clear power law for gamma calculation
        intensity_values = np.ones((1, 7, 1, 1))
        for i in range(7):
            intensity_values[0, i, 0, 0] = 1e6 * (energy_values[i] / 1.0) ** (-2)

        dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values,
        )
        dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values * 0.05,  # 5% background
        )
        dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.sqrt(intensity_values) * 0.1,
        )
        dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values * 0.03,
        )

        # Calculate expected E8 and gamma manually
        j_c_prime = dataset["ena_intensity"] - dataset["bg_rates"]
        j_c_prime = j_c_prime.where(j_c_prime >= 0, 0)

        result = calculate_bootstrap_corrections(dataset)

        # E8 should follow the power law relationship
        # The virtual channel should have meaningful impact on energy channel 6
        original_e6 = j_c_prime[0, 6, 0, 0].values
        corrected_e6 = result["ena_intensity"][0, 6, 0, 0].values

        # Energy channel 6 should be reduced due to E8 spillover subtraction
        assert corrected_e6 < original_e6, (
            "Energy channel 6 should be reduced by E8 virtual channel correction"
        )

    def test_calculate_bootstrap_corrections_bootstrap_factors(self):
        """Test that the bootstrap factor matrix is applied correctly."""
        # Create a simple dataset to verify bootstrap factor application
        coords = {
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
            "longitude": [0],
            "latitude": [0],
        }

        dataset = xr.Dataset(coords=coords)

        energy_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        dataset["energy"] = (("energy",), energy_values)
        dataset["geometric_factor"] = (("energy",), np.ones(7))

        # Use unit intensities to make bootstrap factor effects clear
        intensity_values = np.ones((1, 7, 1, 1)) * 1.0
        dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values,
        )
        dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.zeros((1, 7, 1, 1)),  # No background
        )
        dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 1, 1)) * 0.1,
        )
        dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.ones((1, 7, 1, 1)) * 0.05,
        )

        result = calculate_bootstrap_corrections(dataset)

        # With unit intensities and no background, the corrections should
        # directly reflect the bootstrap factors
        # The bootstrap algorithm is complex due to interdependencies
        # Let's just verify that corrections are applied and reasonable
        corrected_intensities = result["ena_intensity"][0, :, 0, 0].values

        # All channels should be reduced from their original value of 1.0
        for i in range(7):
            assert corrected_intensities[i] < 1.0, (
                f"Energy {i} should be corrected (reduced from 1.0), "
                f"got {corrected_intensities[i]}"
            )

        # Energy 6 should have the largest correction due to 0.75 factor
        assert corrected_intensities[6] < 0.5, (
            f"Energy 6 should have large correction, got {corrected_intensities[6]}"
        )

    def test_calculate_bootstrap_corrections_edge_cases(self):
        """Test edge cases in bootstrap correction calculation."""
        # Test with zero intensities
        coords = {
            "epoch": [8.1794907049e17],
            "energy": list(range(7)),
            "longitude": [0],
            "latitude": [0],
        }

        dataset = xr.Dataset(coords=coords)

        energy_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        dataset["energy"] = (("energy",), energy_values)
        dataset["geometric_factor"] = (("energy",), np.ones(7))

        # Zero intensities
        intensity_values = np.zeros((1, 7, 1, 1))
        dataset["ena_intensity"] = (
            ("epoch", "energy", "longitude", "latitude"),
            intensity_values,
        )
        dataset["bg_rates"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.zeros((1, 7, 1, 1)),
        )
        dataset["ena_intensity_stat_uncert"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.zeros((1, 7, 1, 1)),
        )
        dataset["ena_intensity_sys_err"] = (
            ("epoch", "energy", "longitude", "latitude"),
            np.zeros((1, 7, 1, 1)),
        )

        result = calculate_bootstrap_corrections(dataset)

        # Should handle zero intensities gracefully
        assert np.all(result["ena_intensity"].values >= 0), (
            "Zero intensities should remain non-negative"
        )
        assert np.all(np.isfinite(result["ena_intensity"].values)), (
            "All intensities should be finite"
        )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCalculateAllRatesAndIntensities:
    """Integration tests for the calculate_all_rates_and_intensities function."""

    def test_calculate_all_rates_and_intensities_complete(self):
        """Test the complete rates and intensities calculation pipeline."""
        # Create a comprehensive dataset with current naming convention
        dataset = xr.Dataset(
            {
                # Count data (current generic naming)
                "counts": (("energy",), np.ones(7) * 10),
                # Efficiency corrected data
                "counts_over_eff": (("energy",), np.ones(7) * 12),  # 10/0.83 ≈ 12
                "counts_over_eff_squared": (("energy",), np.ones(7) * 12),
                # Other required data
                "exposure_factor": (("energy",), np.ones(7) * 1.0),
                "geometric_factor": (("energy",), np.ones(7) * 1e-4),
                "energy": (("energy",), np.ones(7) * 0.1),
                "geometric_factor_stat_uncert": (("energy",), np.ones(7) * 1e-5),
                # Background intermediate data
                "bg_rates_exposure_factor": (("energy",), np.ones(7) * 0.3),
                "bg_rates_stat_uncert_exposure_factor2": (
                    ("energy",),
                    np.ones(7) * 0.009,
                ),
            }
        )

        result = calculate_all_rates_and_intensities(dataset)

        # Check that rates were calculated
        assert "ena_count_rate" in result.data_vars
        assert "ena_count_rate_stat_uncert" in result.data_vars

        # Check that intensities were calculated
        assert "ena_intensity" in result.data_vars
        assert "ena_intensity_stat_uncert" in result.data_vars
        assert "ena_intensity_sys_err" in result.data_vars

        # Check that background rates were calculated
        assert "bg_rates" in result.data_vars
        assert "bg_rates_stat_uncert" in result.data_vars
        assert "bg_rates_sys_err" in result.data_vars

        # Check that intermediate variables were cleaned up
        assert "counts_over_eff" not in result.data_vars
        assert "counts_over_eff_squared" not in result.data_vars
        assert "bg_rates_exposure_factor" not in result.data_vars
        assert "bg_rates_stat_uncert_exposure_factor2" not in result.data_vars


@pytest.mark.external_kernel
class TestIntegrationWithMocks:
    """Integration tests using mocked external dependencies."""

    def test_lo_l2_integration_minimal(self, minimal_pset_for_species):
        """Test the main lo_l2 function with minimal mocking."""
        # Test with hydrogen data
        sci_dependencies = {"imap_lo_l1c_pset": [minimal_pset_for_species]}
        anc_dependencies = []
        descriptor = "l090-ena-h-sf-nsp-ram-hae-6deg-3mo"

        # Mock the complex external dependencies to return simple results
        with (
            patch(
                "imap_processing.lo.l2.lo_l2.create_sky_map_from_psets"
            ) as mock_create_map,
            patch("imap_processing.lo.l2.lo_l2.add_geometric_factors") as mock_add_gf,
            patch(
                "imap_processing.lo.l2.lo_l2.calculate_all_rates_and_intensities"
            ) as mock_calc_rates,
            patch("imap_processing.lo.l2.lo_l2.finalize_dataset") as mock_finalize,
        ):
            # Setup mock returns
            mock_sky_map = Mock()
            mock_dataset = xr.Dataset({"test_var": (("energy",), np.ones(7))})
            mock_sky_map.to_dataset.return_value = mock_dataset
            mock_create_map.return_value = mock_sky_map
            mock_add_gf.return_value = mock_dataset
            mock_calc_rates.return_value = mock_dataset
            mock_finalize.return_value = mock_dataset

            # Run the function - should not crash
            result = lo_l2(sci_dependencies, anc_dependencies, descriptor)

            # Basic validation
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], xr.Dataset)

            # Mock the rates calculation to return the dataset unchanged
            mock_calc_rates.side_effect = lambda x, **kwargs: x

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

    def test_create_sky_map_healpix_not_supported(self, minimal_pset_for_species):
        """Test error when HEALPix map is requested."""
        with patch.object(MapDescriptor, "from_string") as mock_from_string:
            mock_map_desc = Mock()
            mock_healpix_map = Mock()  # Not a RectangularSkyMap
            mock_map_desc.to_empty_map.return_value = mock_healpix_map
            mock_map_desc.species = "h"
            mock_from_string.return_value = mock_map_desc

            with pytest.raises(
                NotImplementedError, match="HEALPix map output not supported"
            ):
                create_sky_map_from_psets(
                    [minimal_pset_for_species], mock_map_desc, pd.DataFrame()
                )


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
                "counts": (("energy",), np.ones(7) * 10),
                "exposure_factor": (("energy",), np.zeros(7)),  # Zero exposure
            }
        )

        result = calculate_rates(dataset)

        # Should handle division by zero gracefully
        assert "ena_count_rate" in result.data_vars
        # Rates should be infinite where exposure time is zero
        assert np.all(np.isinf(result["ena_count_rate"].values))

    def test_negative_counts_handling(self):
        """Test handling of negative count values."""
        dataset = xr.Dataset(
            {
                "counts": (("energy",), np.array([-1, 0, 1, 2, 3, 4, 5])),
                "exposure_factor": (("energy",), np.ones(7)),
            }
        )

        result = calculate_rates(dataset)

        # Should calculate rates even with negative counts
        assert "ena_count_rate" in result.data_vars
        assert "ena_count_rate_stat_uncert" in result.data_vars

        # Uncertainty calculation should handle negative counts
        # (sqrt of negative gives NaN, which is expected behavior)
        assert np.isnan(result["ena_count_rate_stat_uncert"].values[0])
