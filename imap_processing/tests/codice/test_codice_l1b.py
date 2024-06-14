"""Tests the L1b processing for CoDICE L1a data"""

from pathlib import Path

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.codice.codice_l1b import process_codice_l1b

EXPECTED_FILENAMES = [
    "imap_codice_l1b_hskp_20100101_v001.cdf",
    "imap_codice_l1b_hi-counters-aggregated_20240429_v001.cdf",
    "imap_codice_l1b_hi-counters-singles_20240429_v001.cdf",
    "imap_codice_l1b_hi-omni_20240429_v001.cdf",
    "imap_codice_l1b_hi-sectored_20240429_v001.cdf",
    "imap_codice_l1b_lo-counters-aggregated_20240429_v001.cdf",
    "imap_codice_l1b_lo-counters-singles_20240429_v001.cdf",
    "imap_codice_l1b_lo-sw-angular_20240429_v001.cdf",
    "imap_codice_l1b_lo-nsw-angular_20240429_v001.cdf",
    "imap_codice_l1b_lo-sw-priority_20240429_v001.cdf",
    "imap_codice_l1b_lo-nsw-priority_20240429_v001.cdf",
    "imap_codice_l1b_lo-sw-species_20240429_v001.cdf",
    "imap_codice_l1b_lo-nsw-species_20240429_v001.cdf",
]
TEST_FILES = [
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_hskp_20100101_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_hi-counters-aggregated_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_hi-counters-singles_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_hi-omni_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_hi-sectored_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_lo-counters-aggregated_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_lo-counters-singles_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_lo-sw-angular_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_lo-nsw-angular_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_lo-sw-priority_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_lo-nsw-priority_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_lo-sw-species_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l1a_lo-nsw-species_20240429_v001.pkts"
    ),
]


@pytest.fixture(params=TEST_FILES)
def test_l1b_data(request) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """

    dataset = process_codice_l1b(request.param, data_version="001")
    return dataset


@pytest.mark.parametrize(
    "test_l1b_data, expected_filename",
    list(zip(TEST_FILES, EXPECTED_FILENAMES)),
    indirect=["test_l1b_data"],
)
def test_l1b_cdf_filenames(test_l1b_data: xr.Dataset, expected_filename: str):
    """Tests that the ``process_codice_l1b`` function generates CDF files with
    expected filenames.

    Parameters
    ----------
    test_l1b_data : xr.Dataset
        A ``xarray`` dataset containing the test data
    expected_filename : str
        The expected CDF filename
    """

    dataset = test_l1b_data
    assert dataset.cdf_filename.name == expected_filename
