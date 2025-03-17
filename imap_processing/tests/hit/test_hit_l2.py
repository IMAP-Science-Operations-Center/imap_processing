import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b.hit_l1b import (
    PARTICLE_ENERGY_RANGE_MAPPING,
    hit_l1b,
    process_summed_rates_data,
)
from imap_processing.hit.l2.hit_l2 import hit_l2, process_summed_flux_data


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
def l1b_summed_rates_dataset(sci_packet_filepath):
    """Get L1B summed rates dataset to test l2 processing function"""
    l1a_datasets = hit_l1a.hit_l1a(sci_packet_filepath, "001")
    for dataset in l1a_datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1a_counts":
            # Calculate livetime from the livetime counter
            livetime = dataset["livetime_counter"] / 270
            summed_rates = process_summed_rates_data(dataset, livetime)
            return summed_rates


def test_process_summed_flux_data(l1b_summed_rates_dataset):
    """Test the variables in the summed flux dataset"""

    l2_summed_flux_dataset = process_summed_flux_data(l1b_summed_rates_dataset)

    # Check that a xarray dataset is returned
    assert isinstance(l2_summed_flux_dataset, xr.Dataset)

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
    assert valid_coords == set(l2_summed_flux_dataset.coords), "Coordinates mismatch"

    assert "dynamic_threshold_state" in l1b_summed_rates_dataset.data_vars

    for particle in PARTICLE_ENERGY_RANGE_MAPPING.keys():
        assert f"{particle}" in l2_summed_flux_dataset.data_vars
        assert f"{particle}_delta_minus" in l2_summed_flux_dataset.data_vars
        assert f"{particle}_delta_plus" in l2_summed_flux_dataset.data_vars
        assert f"{particle}_energy_min" in l2_summed_flux_dataset.data_vars
        assert f"{particle}_energy_max" in l2_summed_flux_dataset.data_vars


def test_hit_l2(dependencies):
    """Test creating L2 datasets ready for CDF output

    Creates a list of xarray datasets for L2 products.

    Parameters
    ----------
    dependencies : dict
        Dictionary of L1B datasets
    """
    # TODO: update assertions after science data processing is completed
    l1b_summed_dataset = dependencies["imap_hit_l1b_summed-rates"]
    datasets = hit_l2(l1b_summed_dataset, "001")
    assert len(datasets) == 1
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l2_summed-intensity"
