import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b.hit_l1b import (
    SUMMED_PARTICLE_ENERGY_RANGE_MAPPING,
    hit_l1b,
)
from imap_processing.hit.l2.hit_l2 import (
    FILLVAL_FLOAT32,
    N_AZIMUTH,
    SECONDS_PER_10_MIN,
    SECONDS_PER_MIN,
    STANDARD_PARTICLE_ENERGY_RANGE_MAPPING,
    VALID_SECTORED_SPECIES,
    add_systematic_uncertainties,
    add_total_uncertainties,
    build_ancillary_dataset,
    calculate_intensities,
    calculate_intensities_for_a_species,
    calculate_intensities_for_all_species,
    get_species_ancillary_data,
    hit_l2,
    process_sectored_intensity_data,
    process_standard_intensity_data,
    process_summed_intensity_data,
    reshape_for_sectored,
)


@pytest.fixture(scope="module")
def sci_packet_filepath():
    """Set path to test data file"""
    return imap_module_directory / "tests/hit/test_data/sci_sample.ccsds"


@pytest.fixture
def dependencies(sci_packet_filepath):
    """Get dependencies for L2 processing"""
    # Create dictionary of dependencies
    data_dict = {}
    l1a_datasets = hit_l1a.hit_l1a(sci_packet_filepath)
    for l1a_dataset in l1a_datasets:
        l1a_data_dict = {}
        if l1a_dataset.attrs["Logical_source"] == "imap_hit_l1a_counts":
            l1a_data_dict["imap_hit_l1a_counts"] = l1a_dataset
            l1b_datasets = hit_l1b(l1a_data_dict)
            for l1b_dataset in l1b_datasets:
                data_dict[l1b_dataset.attrs["Logical_source"]] = l1b_dataset
    return data_dict


@pytest.fixture
def l1b_summed_rates_dataset(dependencies):
    """Get L1B summed rates dataset to test l2 processing function"""
    return dependencies["imap_hit_l1b_summed-rates"]


@pytest.fixture
def l1b_standard_rates_dataset(dependencies):
    """Get L1B standard rates dataset to test l2 processing function"""
    return dependencies["imap_hit_l1b_standard-rates"]


@pytest.fixture
def l1b_sectored_rates_dataset(dependencies):
    """Get L1B standard rates dataset to test l2 processing function"""
    return dependencies["imap_hit_l1b_sectored-rates"]


def _check_ancillary_dataset(
    ancillary_ds,
    species_array,
    delta_e,
    geometry_factors,
    efficiencies,
    b,
    expected_delta_time,
):
    # Helper function - not a test

    shape = species_array.shape

    # Assert that all expected variables are present
    assert set(ancillary_ds.data_vars) == {
        "delta_e",
        "geometry_factor",
        "efficiency",
        "b",
        "delta_time",
    }

    # Check that shapes match
    assert ancillary_ds["delta_e"].shape == shape
    assert ancillary_ds["delta_time"].shape == (len(species_array.coords["epoch"]),)

    # Check values
    np.testing.assert_array_equal(ancillary_ds["delta_e"].values, delta_e)
    np.testing.assert_array_equal(
        ancillary_ds["geometry_factor"].values, geometry_factors
    )
    np.testing.assert_array_equal(ancillary_ds["efficiency"].values, efficiencies)
    np.testing.assert_array_equal(ancillary_ds["b"].values, b)
    np.testing.assert_array_equal(
        ancillary_ds["delta_time"].values,
        np.full(len(species_array.epoch), expected_delta_time),
    )

    # Check coordinates match
    for coord in species_array.coords:
        assert coord in ancillary_ds.coords
        np.testing.assert_array_equal(
            species_array.coords[coord], ancillary_ds.coords[coord]
        )


def test_build_ancillary_dataset_sectored():
    """
    Test the build_ancillary_dataset function for sectored data
    """
    np.random.seed(42)  # Set a random seed for reproducibility
    epoch = np.array(["2025-01-01T00:00", "2025-01-01T00:01"], dtype="datetime64[m]")
    energy_mean = [1.8, 4, 6]
    declination = np.arange(8)
    azimuth = np.arange(15)

    species_array = xr.DataArray(
        data=np.random.rand(2, 3, 15, 8),  # (epoch, energy_mean, azimuth, declination)
        dims=("epoch", "energy_mean", "azimuth", "declination"),
        coords={
            "epoch": epoch,
            "energy_mean": energy_mean,
            "declination": declination,
            "azimuth": azimuth,
        },
        name="h",
    )

    shape = species_array.shape
    delta_e = np.full(shape, 1.0)
    geometry_factors = np.full(shape, 2.0)
    efficiencies = np.full(shape, 0.5)
    b = np.full(shape, 0.1)

    ancillary_ds = build_ancillary_dataset(
        delta_e, geometry_factors, efficiencies, b, species_array
    )
    _check_ancillary_dataset(
        ancillary_ds,
        species_array,
        delta_e,
        geometry_factors,
        efficiencies,
        b,
        SECONDS_PER_10_MIN,
    )


def test_build_ancillary_dataset_nonsectored():
    """
    Test the build_ancillary_dataset function for non-sectored data.

    Non-sectored datasets are either L2 standard or L2 summed datasets
    They both have the same shape (epoch, energy_mean).
    """
    np.random.seed(42)  # Set a random seed for reproducibility
    epoch = np.array(["2025-01-01T00:00", "2025-01-01T00:01"], dtype="datetime64[m]")
    energy_mean = [1.8, 4, 6]

    species_array = xr.DataArray(
        data=np.random.rand(2, 3),  # (epoch, energy_mean)
        dims=("epoch", "energy_mean"),
        coords={"epoch": epoch, "energy_mean": energy_mean},
        name="h",
    )

    shape = species_array.shape
    delta_e = np.full(shape, 1.0)
    geometry_factors = np.full(shape, 2.0)
    efficiencies = np.full(shape, 0.5)
    b = np.full(shape, 0.1)

    ancillary_ds = build_ancillary_dataset(
        delta_e, geometry_factors, efficiencies, b, species_array
    )
    _check_ancillary_dataset(
        ancillary_ds,
        species_array,
        delta_e,
        geometry_factors,
        efficiencies,
        b,
        SECONDS_PER_MIN,
    )


def test_get_species_ancillary_data():
    """Test the get_species_ancillary_data function."""

    # Mock ancillary data for dynamic threshold states 0 and 1
    ancillary_data_frames = {
        0: pd.DataFrame(
            {
                "species": ["h", "h", "he", "he"],
                "lower energy (mev)": [1, 2, 1, 2],
                "delta e (mev)": [0.1, 0.2, 0.3, 0.4],
                "geometry factor (cm2 sr)": [10, 20, 30, 40],
                "efficiency": [0.9, 0.8, 0.7, 0.6],
                "b": [0.01, 0.02, 0.03, 0.04],
            }
        ),
        1: pd.DataFrame(
            {
                "species": ["h", "h", "he", "he"],
                "lower energy (mev)": [1, 2, 1, 2],
                "delta e (mev)": [0.15, 0.25, 0.35, 0.45],
                "geometry factor (cm2 sr)": [15, 25, 35, 45],
                "efficiency": [0.85, 0.75, 0.65, 0.55],
                "b": [0.015, 0.025, 0.035, 0.045],
            }
        ),
    }

    # Test for dynamic threshold state 0 and species "h"
    result = get_species_ancillary_data(0, ancillary_data_frames, "h")
    expected = {
        "delta_e": np.array([[0.1], [0.2]]),
        "geometry_factor": np.array([[10], [20]]),
        "efficiency": np.array([[0.9], [0.8]]),
        "b": np.array([[0.01], [0.02]]),
    }
    for key, value in expected.items():
        np.testing.assert_array_equal(result[key], value)

    # Test for dynamic threshold state 1 and species "he"
    result = get_species_ancillary_data(1, ancillary_data_frames, "he")
    expected = {
        "delta_e": np.array([[0.35], [0.45]]),
        "geometry_factor": np.array([[35], [45]]),
        "efficiency": np.array([[0.65], [0.55]]),
        "b": np.array([[0.035], [0.045]]),
    }
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])


def test_reshape_for_sectored():
    """
    Test the reshape_for_sectored function.
    """
    # Mock input data: 3D array (epoch, energy, declination)
    np.random.seed(42)  # Set a random seed for reproducibility
    epoch, energy, declination = 2, 3, 8
    input_array = np.random.rand(epoch, energy, declination)

    # Expected output shape: 4D array (epoch, energy, azimuth, declination)
    expected_shape = (epoch, energy, N_AZIMUTH, declination)

    # Call the function
    reshaped_array = reshape_for_sectored(input_array)

    # Assertions
    assert reshaped_array.shape == expected_shape, "Output shape mismatch"
    for azimuth in range(N_AZIMUTH):
        np.testing.assert_array_equal(
            reshaped_array[:, :, azimuth, :],
            input_array,
            err_msg=f"Mismatch in azimuth dimension {azimuth}",
        )


def test_calculate_intensities_for_all_species():
    """Test the calculate_intensities_for_all_species function."""
    # Sample input data
    l2_dataset = xr.Dataset(
        {
            "dynamic_threshold_state": ("epoch", np.array([0, 1])),
            "h": (
                ("epoch", "h_energy_mean"),
                np.array([[100, 200, 300], [400, 500, 600]]).astype("float32"),
            ),
            "h_energy_mean": (
                "h_energy_mean",
                np.array([1.0, 2.0, 3.0]).astype("float32"),
            ),
            "h_energy_delta_minus": (
                "h_energy_mean",
                np.array([0.1, 0.1, 0.1]).astype("float32"),
            ),
            "h_energy_delta_plus": (
                "h_energy_mean",
                np.array([0.1, 0.1, 0.1]).astype("float32"),
            ),
            "ni": (
                ("epoch", "ni_energy_mean"),
                np.array([[150, 250, 350], [450, 550, 650]]).astype("float32"),
            ),
            "ni_energy_mean": (
                "ni_energy_mean",
                np.array([1.5, 2.5, 3.5]).astype("float32"),
            ),
            "ni_energy_delta_minus": (
                "ni_energy_mean",
                np.array([0.1, 0.1, 0.1]).astype("float32"),
            ),
            "ni_energy_delta_plus": (
                "ni_energy_mean",
                np.array([0.1, 0.1, 0.1]).astype("float32"),
            ),
        }
    )
    ancillary_data_frames = {
        0: pd.DataFrame(
            {
                "species": ["h", "h", "h", "ni", "ni", "ni"],
                "lower energy (mev)": [0.9, 1.9, 2.9, 1.4, 2.4, 3.4],
                "delta e (mev)": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "geometry factor (cm2 sr)": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "efficiency": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "b": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        1: pd.DataFrame(
            {
                "species": ["h", "h", "h", "ni", "ni", "ni"],
                "lower energy (mev)": [0.9, 1.9, 2.9, 1.4, 2.4, 3.4],
                "delta e (mev)": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "geometry factor (cm2 sr)": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "efficiency": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "b": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
    }

    # Expected output
    expected_intensities_h = xr.DataArray(
        [[8.333333, 16.666667, 25.0], [33.333333, 41.666667, 50.0]],
        dims=["epoch", "h_energy_mean"],
    )
    expected_intensities_ni = xr.DataArray(
        [[12.5, 20.833333, 29.166667], [37.5, 45.833333, 54.166667]],
        dims=["epoch", "ni_energy_mean"],
    )

    # Call the function
    l2_dataset = calculate_intensities_for_all_species(
        l2_dataset, ancillary_data_frames, valid_data_variables=["h", "ni"]
    )

    # Assertions
    (
        np.testing.assert_allclose(
            l2_dataset["h"].values, expected_intensities_h.values
        ),
        ("Intensities mismatch for H"),
    )
    (
        np.testing.assert_allclose(
            l2_dataset["ni"].values, expected_intensities_ni.values
        ),
        ("Intensities mismatch for He"),
    )


def test_calculate_intensities_for_a_species():
    """Test the calculate_intensities_for_a_species function."""
    # Sample input data
    species_variable = "h"
    l2_dataset = xr.Dataset(
        {
            "h": (
                ("epoch", "h_energy_mean"),
                np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]).astype(
                    "float32"
                ),
            ),
            "dynamic_threshold_state": ("epoch", np.array([0, 1])),
            "h_energy_mean": (
                "h_energy_mean",
                np.array([1.0, 2.0, 3.0]).astype("float32"),
            ),
            "h_energy_delta_minus": (
                "h_energy_mean",
                np.array([0.1, 0.1, 0.1]).astype("float32"),
            ),
        }
    )
    ancillary_data_frames = {
        0: pd.DataFrame(
            {
                "species": ["h", "h", "h"],
                "lower energy (mev)": [0.9, 1.9, 2.9],
                "delta e (mev)": [0.2, 0.5, 0.5],
                "geometry factor (cm2 sr)": [1.0, 1.0, 1.5],
                "efficiency": [1, 1, 1],
                "b": [0, 0, 0],
            }
        ),
        1: pd.DataFrame(
            {
                "species": ["h", "h", "h"],
                "lower energy (mev)": [0.9, 1.9, 2.9],
                "delta e (mev)": [0.4, 0.5, 0.5],
                "geometry factor (cm2 sr)": [1.5, 1.51, 1.51],
                "efficiency": [1, 1, 1],
                "b": [0, 0, 0],
            }
        ),
    }

    # Expected output
    expected_intensities = xr.DataArray(
        [[8.333333, 6.666667, 6.666667], [11.111111, 11.037528, 13.245033]],
        dims=["epoch", "h_energy_mean"],
    )

    # Call the function
    l2_dataset = calculate_intensities_for_a_species(
        species_variable, l2_dataset, ancillary_data_frames
    )

    # Assertions
    (
        np.testing.assert_allclose(l2_dataset["h"].values, expected_intensities.values),
        ("Intensities mismatch"),
    )


def test_calculate_intensities():
    """Test the calculate_intensities function."""
    # Create sample function inputs
    rates = xr.DataArray(
        data=[10.0, 20.0, FILLVAL_FLOAT32, 40.0],
        dims=["epoch"],
        coords={"epoch": [0, 1, 2, 3]},
    )
    factors = xr.Dataset(
        {
            "delta_time": ("epoch", [60.0, 60.0, 60.0, 60.0]),
            "delta_e": ("epoch", [1.0, 1.0, 1.0, 1.0]),
            "geometry_factor": ("epoch", [2.0, 2.0, 2.0, 2.0]),
            "efficiency": ("epoch", [0.5, 0.5, 0.5, 0.5]),
            "b": ("epoch", [0.0, 0, 0, 0]),
        }
    )

    # Expected output
    expected_intensity = xr.DataArray(
        data=[0.1666667, 0.3333333, FILLVAL_FLOAT32, 0.6666667],
        dims=["epoch"],
        coords={"epoch": [0, 1, 2, 3]},
    )

    # Call the function
    result = calculate_intensities(rates, factors)

    # Assertions
    xr.testing.assert_allclose(result, expected_intensity)


def test_add_systematic_uncertainties():
    """Test the add_systematic_uncertainties function."""
    # Create sample function inputs
    np.random.seed(42)  # Set a random seed for reproducibility
    particle = "h"
    datasets = [
        xr.Dataset(
            {"h": (("epoch", "h_energy_mean"), np.random.rand(2, 3).astype("float32"))}
        ),
        xr.Dataset(
            {
                "h": (
                    ("epoch", "h_energy_mean", "azimuth", "declination"),
                    np.random.rand(2, 3, 15, 8).astype("float32"),
                )
            }
        ),
    ]

    for dataset in datasets:
        # Call the function
        updated_dataset = add_systematic_uncertainties(dataset, particle)
        # Assertions
        assert f"{particle}_sys_err_minus" in updated_dataset.data_vars
        assert f"{particle}_sys_err_plus" in updated_dataset.data_vars
        assert (
            updated_dataset[f"{particle}_sys_err_minus"].shape
            == dataset[particle].shape
        )
        assert (
            updated_dataset[f"{particle}_sys_err_plus"].shape == dataset[particle].shape
        )
        np.testing.assert_array_equal(
            updated_dataset[f"{particle}_sys_err_minus"].values, 0
        )
        np.testing.assert_array_equal(
            updated_dataset[f"{particle}_sys_err_plus"].values, 0
        )


def test_add_total_uncertainties():
    # Create a sample dataset
    np.random.seed(42)  # Set a random seed for reproducibility
    data = np.random.rand(10, 5).astype(np.float32)
    stat_uncert_minus = np.random.rand(10, 5).astype(np.float32)
    stat_uncert_plus = np.random.rand(10, 5).astype(np.float32)
    sys_err_minus = np.zeros(
        (10, 5), dtype=np.float32
    )  # zeros, unless changed during mission
    sys_err_plus = np.zeros(
        (10, 5), dtype=np.float32
    )  # zeros, unless changed during mission

    dataset = xr.Dataset(
        {
            "particle": (("dim_0", "dim_1"), data),
            "particle_stat_uncert_minus": (("dim_0", "dim_1"), stat_uncert_minus),
            "particle_stat_uncert_plus": (("dim_0", "dim_1"), stat_uncert_plus),
            "particle_sys_err_minus": (("dim_0", "dim_1"), sys_err_minus),
            "particle_sys_err_plus": (("dim_0", "dim_1"), sys_err_plus),
        }
    )

    # Call the function
    updated_dataset = add_total_uncertainties(dataset, "particle")

    # Assertions
    np.testing.assert_array_almost_equal(
        updated_dataset["particle_total_uncert_minus"].values,
        np.sqrt(np.square(stat_uncert_minus) + np.square(sys_err_minus)),
    )
    np.testing.assert_array_almost_equal(
        updated_dataset["particle_total_uncert_plus"].values,
        np.sqrt(np.square(stat_uncert_plus) + np.square(sys_err_plus)),
    )

    # Check that the dimensions and attributes are preserved
    assert (
        updated_dataset["particle_total_uncert_minus"].dims == dataset["particle"].dims
    )
    assert (
        updated_dataset["particle_total_uncert_plus"].dims == dataset["particle"].dims
    )


def test_process_sectored_intensity_data(l1b_sectored_rates_dataset):
    """Test the variables in the sectored intensity dataset"""

    l2_sectored_intensity_dataset = process_sectored_intensity_data(
        l1b_sectored_rates_dataset
    )

    # Check that a xarray dataset is returned
    assert isinstance(l2_sectored_intensity_dataset, xr.Dataset)

    valid_coords = {
        "epoch",
        "azimuth",
        "declination",
        "h_energy_mean",
        "he4_energy_mean",
        "cno_energy_mean",
        "nemgsi_energy_mean",
        "fe_energy_mean",
    }

    # Check that the dataset has the correct coords and variables
    assert valid_coords == set(l2_sectored_intensity_dataset.coords), (
        "Coordinates mismatch"
    )

    assert "dynamic_threshold_state" in l2_sectored_intensity_dataset.data_vars

    for particle in VALID_SECTORED_SPECIES:
        assert f"{particle}" in l2_sectored_intensity_dataset.data_vars
        assert (
            f"{particle}_stat_uncert_minus" in l2_sectored_intensity_dataset.data_vars
        )
        assert f"{particle}_stat_uncert_plus" in l2_sectored_intensity_dataset.data_vars
        assert f"{particle}_sys_err_minus" in l2_sectored_intensity_dataset.data_vars
        assert f"{particle}_sys_err_plus" in l2_sectored_intensity_dataset.data_vars
        assert (
            f"{particle}_energy_delta_minus" in l2_sectored_intensity_dataset.data_vars
        )
        assert (
            f"{particle}_energy_delta_plus" in l2_sectored_intensity_dataset.data_vars
        )


def test_process_summed_intensity_data(l1b_summed_rates_dataset):
    """Test the variables in the summed intensity dataset"""

    l2_summed_intensity_dataset = process_summed_intensity_data(
        l1b_summed_rates_dataset
    )

    # Check that a xarray dataset is returned
    assert isinstance(l2_summed_intensity_dataset, xr.Dataset)

    valid_coords = {
        "epoch",
        "h_energy_mean",
        "he3_energy_mean",
        "he4_energy_mean",
        "he_energy_mean",
        "c_energy_mean",
        "o_energy_mean",
        "fe_energy_mean",
        "n_energy_mean",
        "si_energy_mean",
        "mg_energy_mean",
        "s_energy_mean",
        "ar_energy_mean",
        "ca_energy_mean",
        "na_energy_mean",
        "al_energy_mean",
        "ne_energy_mean",
        "ni_energy_mean",
    }

    # Check that the dataset has the correct coords and variables
    assert valid_coords == set(l2_summed_intensity_dataset.coords), (
        "Coordinates mismatch"
    )

    assert "dynamic_threshold_state" in l1b_summed_rates_dataset.data_vars

    for particle in SUMMED_PARTICLE_ENERGY_RANGE_MAPPING.keys():
        assert f"{particle}" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_stat_uncert_minus" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_stat_uncert_plus" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_sys_err_minus" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_sys_err_plus" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_energy_delta_minus" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_energy_delta_plus" in l2_summed_intensity_dataset.data_vars


def test_process_standard_intensity_data(l1b_standard_rates_dataset):
    """Test the variables in the standard intensity dataset"""

    l2_standard_intensity_dataset = process_standard_intensity_data(
        l1b_standard_rates_dataset
    )

    # Check that a xarray dataset is returned
    assert isinstance(l2_standard_intensity_dataset, xr.Dataset)

    valid_coords = {
        "epoch",
        "h_energy_mean",
        "he3_energy_mean",
        "he4_energy_mean",
        "he_energy_mean",
        "c_energy_mean",
        "n_energy_mean",
        "o_energy_mean",
        "ne_energy_mean",
        "na_energy_mean",
        "mg_energy_mean",
        "al_energy_mean",
        "si_energy_mean",
        "s_energy_mean",
        "ar_energy_mean",
        "ca_energy_mean",
        "fe_energy_mean",
        "ni_energy_mean",
    }

    # Check that the dataset has the correct coords and variables
    assert valid_coords == set(l2_standard_intensity_dataset.coords), (
        "Coordinates mismatch"
    )

    assert "dynamic_threshold_state" in l1b_standard_rates_dataset.data_vars

    for particle in STANDARD_PARTICLE_ENERGY_RANGE_MAPPING.keys():
        assert f"{particle}" in l2_standard_intensity_dataset.data_vars
        assert (
            f"{particle}_stat_uncert_minus" in l2_standard_intensity_dataset.data_vars
        )
        assert f"{particle}_stat_uncert_plus" in l2_standard_intensity_dataset.data_vars
        assert f"{particle}_sys_err_minus" in l2_standard_intensity_dataset.data_vars
        assert f"{particle}_sys_err_plus" in l2_standard_intensity_dataset.data_vars
        assert (
            f"{particle}_energy_delta_minus" in l2_standard_intensity_dataset.data_vars
        )
        assert (
            f"{particle}_energy_delta_plus" in l2_standard_intensity_dataset.data_vars
        )


def test_hit_l2(dependencies):
    """Test creating L2 datasets ready for CDF output

    Creates a list of xarray datasets for L2 products.

    Parameters
    ----------
    dependencies : dict
        Dictionary of L1B datasets
    """
    # TODO: update assertions after science data processing is completed
    l2_datasets = hit_l2(dependencies["imap_hit_l1b_summed-rates"])
    assert len(l2_datasets) == 1
    assert l2_datasets[0].attrs["Logical_source"] == "imap_hit_l2_summed-intensity"

    l2_datasets = hit_l2(dependencies["imap_hit_l1b_standard-rates"])
    assert len(l2_datasets) == 1
    assert l2_datasets[0].attrs["Logical_source"] == "imap_hit_l2_standard-intensity"

    l2_datasets = hit_l2(dependencies["imap_hit_l1b_sectored-rates"])
    assert len(l2_datasets) == 1
    assert l2_datasets[0].attrs["Logical_source"] == "imap_hit_l2_macropixel-intensity"
