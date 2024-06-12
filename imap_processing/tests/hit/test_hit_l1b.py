import pathlib

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.hit.l1b import hit_l1b


@pytest.fixture(scope="module")
def dependency():
    """Set path to test data file"""

    cdf_filepath = (
        imap_module_directory / "tests/hit/test_data/imap_hit_l1a_hk_20100105_v001.cdf"
    )
    l1a_data = load_cdf(cdf_filepath)
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
    cdf_filepaths = hit_l1b.hit_l1b(dependency)
    assert len(cdf_filepaths) == 1
    assert isinstance(cdf_filepaths[0], pathlib.PurePath)
    assert cdf_filepaths[0].name == "imap_hit_l1b_hk_20100101_v001.cdf"
