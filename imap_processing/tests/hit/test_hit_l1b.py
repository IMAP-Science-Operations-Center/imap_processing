import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b import hit_l1b


@pytest.fixture(scope="module")
def packet_filepath():
    """Set path to test data file"""
    return (
        imap_module_directory / "tests/hit/test_data/imap_hit_l0_raw_20100105_v001.pkts"
    )


@pytest.fixture()
def dependencies(packet_filepath):
    """Get dependencies for L1B processing"""
    # Create dictionary of dependencies and add CCSDS packet file
    data_dict = {"imap_hit_l0_raw": packet_filepath}
    # Add L1A datasets
    l1a_datasets = hit_l1a.hit_l1a(packet_filepath, "001")
    for dataset in l1a_datasets:
        data_dict[dataset.attrs["Logical_source"]] = dataset
    return data_dict


def test_create_l1b_hk_dataset(packet_filepath):
    """Test creating housekeeping L1B dataset

    Creates a xarray dataset for housekeeping data
    """
    datasets = hit_l1b.create_l1b_hk_dataset(packet_filepath, "001")
    assert len(datasets) == 1
    assert isinstance(datasets[0], xr.Dataset)
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l1b_hk"


def test_hit_l1b(dependencies):
    """Test creating L1B CDF files

    Creates a list of xarray datasets for each L1B product

    Parameters
    ----------
    dependencies : dict
        Dictionary of L1A datasets and CCSDS packet file path
    """
    # TODO: update assertions after science data processing is completed
    datasets = hit_l1b.hit_l1b(dependencies, "001")
    assert len(datasets) == 1
    assert isinstance(datasets[0], xr.Dataset)
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l1b_hk"
