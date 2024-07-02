import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b import hit_l1b


@pytest.fixture()
def dependency():
    """Get L1A data from test packet file"""

    packet_filepath = imap_module_directory / "tests/hit/test_data/hskp_sample.ccsds"
    l1a_data = hit_l1a.hit_l1a(packet_filepath, "001")[0]

    return l1a_data


def test_create_hk_dataset():
    """Test creating housekeeping L1B dataset

    Creates a xarray dataset for housekeeping data
    """

    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hit")
    attr_mgr.add_instrument_variable_attrs(instrument="hit", level="l1b")
    attr_mgr.add_global_attribute("Data_version", "001")

    l1b_hk_dataset = hit_l1b.create_hk_dataset(attr_mgr)
    assert isinstance(l1b_hk_dataset, xr.Dataset)


def test_hit_l1b(dependency):
    """Test creating L1B CDF files

    Creates a CDF file for each L1B product and stores
    their filepaths in a list

    Parameters
    ----------
    dependency : xr.dataset
        L1A data
    """
    datasets = hit_l1b.hit_l1b(dependency, "001")
    assert len(datasets) == 1
    assert isinstance(datasets[0], xr.Dataset)
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l1b_hk"
