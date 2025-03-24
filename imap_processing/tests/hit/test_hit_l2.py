import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b.hit_l1b import (
    PARTICLE_ENERGY_RANGE_MAPPING,
    hit_l1b,
)
from imap_processing.hit.l2.hit_l2 import (
    STANDARD_PARTICLE_ENERGY_RANGE_MAPPING,
    IntensityFactors,
    add_systematic_uncertainties,
    calculate_intensities,
    calculate_intensity_for_a_species,
    calculate_intensity_for_all_species,
    get_intensity_factors,
    get_species_ancillary_data,
    hit_l2,
    process_standard_intensity_data,
    process_summed_intensity_data,
)

# TODO: add unit test for add_standard_particle_rates_to_dataset


@pytest.fixture(scope="module")
def sci_packet_filepath():
    """Set path to test data file"""
    return imap_module_directory / "tests/hit/test_data/sci_sample.ccsds"


@pytest.fixture()
def dependencies(sci_packet_filepath):
    """Get dependencies for L2 processing"""
    # Create dictionary of dependencies
    data_dict = {}
    l1a_datasets = hit_l1a.hit_l1a(sci_packet_filepath, "001")
    for l1a_dataset in l1a_datasets:
        l1a_data_dict = {}
        if l1a_dataset.attrs["Logical_source"] == "imap_hit_l1a_counts":
            l1a_data_dict["imap_hit_l1a_counts"] = l1a_dataset
            l1b_datasets = hit_l1b(l1a_data_dict, "001")
            for l1b_dataset in l1b_datasets:
                data_dict[l1b_dataset.attrs["Logical_source"]] = l1b_dataset
    return data_dict


@pytest.fixture()
def l1b_summed_rates_dataset(dependencies):
    """Get L1B summed rates dataset to test l2 processing function"""
    return dependencies["imap_hit_l1b_summed-rates"]


@pytest.fixture()
def l1b_standard_rates_dataset(dependencies):
    """Get L1B standard rates dataset to test l2 processing function"""
    return dependencies["imap_hit_l1b_standard-rates"]


def test_get_intensity_factors():
    """Test the get_intensity_factors function."""
    # Sample input data
    energy_min = np.array([1.8, 2.2, 2.7], dtype=np.float32)
    species_ancillary_data = pd.DataFrame(
        {
            "lower energy (mev)": [1.8, 2.2, 2.7],
            "delta e (mev)": [0.4, 0.5, 0.6],
            "geometry factor (cm2 sr)": [1.0, 1.1, 1.2],
            "efficiency": [0.9, 0.8, 0.7],
            "b": [0.1, 0.2, 0.3],
        }
    )

    # Expected output
    expected_factors = IntensityFactors(
        delta_e_factor=np.array([0.4, 0.5, 0.6]),
        geometry_factor=np.array([1.0, 1.1, 1.2]),
        efficiency=np.array([0.9, 0.8, 0.7]),
        b=np.array([0.1, 0.2, 0.3]),
    )

    # Call the function
    factors = get_intensity_factors(energy_min, species_ancillary_data)

    # Assertions
    assert np.array_equal(
        factors.delta_e_factor, expected_factors.delta_e_factor
    ), "Delta E factors mismatch"
    assert np.array_equal(
        factors.geometry_factor, expected_factors.geometry_factor
    ), "Geometry factors mismatch"
    assert np.array_equal(
        factors.efficiency, expected_factors.efficiency
    ), "Efficiency factors mismatch"
    assert np.array_equal(factors.b, expected_factors.b), "B factors mismatch"


def test_get_species_ancillary_data():
    """Test the get_species_ancillary_data function."""
    # Sample input data
    dynamic_threshold_state = 1
    ancillary_data_frames = {
        0: pd.DataFrame(
            {
                "species": ["h", "he", "c"],
                "lower energy (mev)": [1.0, 2.0, 3.0],
                "delta e (mev)": [0.1, 0.2, 0.3],
                "geometry factor (cm2 sr)": [1.0, 1.1, 1.2],
                "efficiency": [0.9, 0.8, 0.7],
                "b": [0.01, 0.02, 0.03],
            }
        ),
        1: pd.DataFrame(
            {
                "species": ["h", "he", "c"],
                "lower energy (mev)": [1.0, 2.0, 3.0],
                "delta e (mev)": [0.15, 0.25, 0.35],
                "geometry factor (cm2 sr)": [1.05, 1.15, 1.25],
                "efficiency": [0.85, 0.75, 0.65],
                "b": [0.015, 0.025, 0.035],
            }
        ),
    }
    species = "h"

    # Expected output
    expected_output = ancillary_data_frames[dynamic_threshold_state][
        ancillary_data_frames[dynamic_threshold_state]["species"] == species
    ]

    # Call the function
    output = get_species_ancillary_data(
        dynamic_threshold_state, ancillary_data_frames, species
    )

    # Assertions
    pd.testing.assert_frame_equal(output, expected_output)


def test_calculate_intensity_for_all_species():
    """Test the calculate_intensity_for_all_species function."""
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
    calculate_intensity_for_all_species(l2_dataset, ancillary_data_frames)

    # Assertions
    assert np.allclose(
        l2_dataset["h"].values, expected_intensities_h.values
    ), "Intensities mismatch for H"
    assert np.allclose(
        l2_dataset["ni"].values, expected_intensities_ni.values
    ), "Intensities mismatch for He"


def test_calculate_intensity_for_a_species():
    """Test the calculate_intensity_for_a_species function."""
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
    calculate_intensity_for_a_species(
        species_variable, l2_dataset, ancillary_data_frames
    )

    # Assertions
    assert np.allclose(
        l2_dataset["h"].values, expected_intensities.values
    ), "Intensities mismatch"


def test_calculate_intensities():
    """Test the calculate_intensities function."""
    # Sample input data
    rate = xr.DataArray([100, 200, 300], dims=["energy_bin"])
    delta_e_factor = np.array([1.0, 1.0, 1.0])
    geometry_factor = np.array([1.0, 1.0, 1.0])
    efficiency = np.array([1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.0])

    # Expected output
    expected_intensities = xr.DataArray(
        [1.66666667, 3.33333333, 5.0], dims=["energy_bin"]
    )

    # Call the function
    intensities = calculate_intensities(
        rate, delta_e_factor, geometry_factor, efficiency, b
    )

    # Assertions
    assert np.allclose(
        intensities.values, expected_intensities.values
    ), "Intensities mismatch"


def test_add_systematic_uncertainties():
    """Test the add_systematic_uncertainties function."""
    # Create sample function inputs
    particle = "h"
    energy_ranges = [
        {"energy_min": 1.8, "energy_max": 2.2, "R2": [1], "R3": [], "R4": []},
        {"energy_min": 2.2, "energy_max": 2.7, "R2": [2], "R3": [], "R4": []},
        {"energy_min": 2.7, "energy_max": 3.2, "R2": [3], "R3": [], "R4": []},
    ]
    dataset = xr.Dataset()

    # Call the function
    add_systematic_uncertainties(dataset, particle, energy_ranges)

    # Assertions
    assert f"{particle}_sys_delta_minus" in dataset.data_vars
    assert f"{particle}_sys_delta_plus" in dataset.data_vars
    assert np.all(dataset[f"{particle}_sys_delta_minus"].values == 0)
    assert np.all(dataset[f"{particle}_sys_delta_plus"].values == 0)
    assert dataset[f"{particle}_sys_delta_minus"].shape == (len(energy_ranges),)
    assert dataset[f"{particle}_sys_delta_plus"].shape == (len(energy_ranges),)


def test_process_summed_intensity_data(l1b_summed_rates_dataset):
    """Test the variables in the summed intensity dataset"""

    l2_summed_intensity_dataset = process_summed_intensity_data(
        l1b_summed_rates_dataset
    )

    # Check that a xarray dataset is returned
    assert isinstance(l2_summed_intensity_dataset, xr.Dataset)

    valid_coords = {
        "epoch",
        "h_energy_index",
        "he3_energy_index",
        "he4_energy_index",
        "he_energy_index",
        "c_energy_index",
        "o_energy_index",
        "fe_energy_index",
        "n_energy_index",
        "si_energy_index",
        "mg_energy_index",
        "s_energy_index",
        "ar_energy_index",
        "ca_energy_index",
        "na_energy_index",
        "al_energy_index",
        "ne_energy_index",
        "ni_energy_index",
    }

    # Check that the dataset has the correct coords and variables
    assert valid_coords == set(
        l2_summed_intensity_dataset.coords
    ), "Coordinates mismatch"

    assert "dynamic_threshold_state" in l1b_summed_rates_dataset.data_vars

    for particle in PARTICLE_ENERGY_RANGE_MAPPING.keys():
        assert f"{particle}" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_delta_minus" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_delta_plus" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_energy_min" in l2_summed_intensity_dataset.data_vars
        assert f"{particle}_energy_max" in l2_summed_intensity_dataset.data_vars


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
    assert valid_coords == set(
        l2_standard_intensity_dataset.coords
    ), "Coordinates mismatch"

    assert "dynamic_threshold_state" in l1b_standard_rates_dataset.data_vars

    for particle in STANDARD_PARTICLE_ENERGY_RANGE_MAPPING.keys():
        assert f"{particle}" in l2_standard_intensity_dataset.data_vars
        assert f"{particle}_delta_minus" in l2_standard_intensity_dataset.data_vars
        assert f"{particle}_delta_plus" in l2_standard_intensity_dataset.data_vars
        assert f"{particle}_sys_delta_minus" in l2_standard_intensity_dataset.data_vars
        assert f"{particle}_sys_delta_plus" in l2_standard_intensity_dataset.data_vars
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
    l2_datasets = hit_l2(dependencies["imap_hit_l1b_summed-rates"], "001")
    assert len(l2_datasets) == 1
    assert l2_datasets[0].attrs["Logical_source"] == "imap_hit_l2_summed-intensity"

    l2_datasets = hit_l2(dependencies["imap_hit_l1b_standard-rates"], "001")
    assert len(l2_datasets) == 1
    assert l2_datasets[0].attrs["Logical_source"] == "imap_hit_l2_standard-intensity"
