"""Tests the L2 processing of CoDICE L1 data"""

import pytest
import xarray as xr

from imap_processing.codice.codice_l2 import process_codice_l2

from .conftest import TEST_L2_FILES

pytestmark = pytest.mark.external_test_data

EXPECTED_LOGICAL_SOURCES = [
    "imap_codice_l2_hi-direct-events",
    "imap_codice_l2_lo-direct-events",
]


@pytest.fixture(params=TEST_L2_FILES)
def test_l2_data(request) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    dataset = process_codice_l2(request.param)
    return dataset


@pytest.mark.parametrize(
    "test_l2_data, expected_logical_source",
    list(zip(TEST_L2_FILES, EXPECTED_LOGICAL_SOURCES)),
    indirect=["test_l2_data"],
)
def test_l2_logical_sources(test_l2_data: xr.Dataset, expected_logical_source: str):
    """Tests that the ``process_codice_l2`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    test_l2_data : xr.Dataset
        A ``xarray`` dataset containing the test data
    expected_logical_source : str
        The expected CDF filename
    """

    dataset = test_l2_data

    assert dataset.attrs["Logical_source"] == expected_logical_source
