"""Tests the L1b processing for CoDICE L1a data"""

import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l1b import process_codice_l1b

from .conftest import TEST_L1A_FILES

EXPECTED_LOGICAL_SOURCE = [
    "imap_codice_l1b_hskp",
    "imap_codice_l1b_hi-counters-aggregated",
    "imap_codice_l1b_hi-counters-singles",
    "imap_codice_l1b_hi-omni",
    "imap_codice_l1b_hi-sectored",
    "imap_codice_l1b_lo-counters-aggregated",
    "imap_codice_l1b_lo-counters-singles",
    "imap_codice_l1b_lo-sw-angular",
    "imap_codice_l1b_lo-nsw-angular",
    "imap_codice_l1b_lo-sw-priority",
    "imap_codice_l1b_lo-nsw-priority",
    "imap_codice_l1b_lo-sw-species",
    "imap_codice_l1b_lo-nsw-species",
]


@pytest.fixture(params=TEST_L1A_FILES)
def test_l1b_data(request) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    input_dataset = load_cdf(request.param)
    dataset = process_codice_l1b(input_dataset, data_version="001")
    return dataset


@pytest.mark.skip("Awaiting proper implementation of L1B")
@pytest.mark.parametrize(
    "test_l1b_data, expected_logical_source",
    list(zip(TEST_L1A_FILES, EXPECTED_LOGICAL_SOURCE)),
    indirect=["test_l1b_data"],
)
def test_l1b_cdf_filenames(test_l1b_data: xr.Dataset, expected_logical_source: str):
    """Tests that the ``process_codice_l1b`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    test_l1b_data : xr.Dataset
        A ``xarray`` dataset containing the test data
    expected_filename : str
        The expected CDF filename
    """

    dataset = test_l1b_data
    assert dataset.attrs["Logical_source"] == expected_logical_source
