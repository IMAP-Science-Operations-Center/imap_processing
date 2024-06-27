"""Tests the L1b processing for CoDICE L1a data"""

from pathlib import Path

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l1b import process_codice_l1b

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
    input_dataset = load_cdf(request.param)
    dataset = process_codice_l1b(input_dataset, data_version="001")
    return dataset


@pytest.mark.parametrize(
    "test_l1b_data, expected_logical_source",
    list(zip(TEST_FILES, EXPECTED_LOGICAL_SOURCE)),
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
