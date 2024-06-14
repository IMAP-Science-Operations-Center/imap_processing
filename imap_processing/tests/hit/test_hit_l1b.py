import pathlib

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b import hit_l1b


@pytest.fixture()
def dependency():
    """Get L1A data from test packet file"""

    packet_filepath = imap_module_directory / "tests/hit/test_data/hskp_sample.ccsds"
    l1a_data = load_cdf(hit_l1a.hit_l1a(packet_filepath, "001")[0])

    return l1a_data


def test_create_hk_dataset():
    """Test creating housekeeping L1B dataset

    Creates a xarray dataset for housekeeping data
    """

    l1b_hk_dataset = hit_l1b.create_hk_dataset()
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
    cdf_filepaths = hit_l1b.hit_l1b(dependency, "001")
    assert len(cdf_filepaths) == 1
    assert isinstance(cdf_filepaths[0], pathlib.PurePath)
    assert cdf_filepaths[0].name == "imap_hit_l1b_hk_20100101_v001.cdf"
