"""Tests the L1a processing for decommutated CoDICE data"""

import logging

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_codice_l1a

from .conftest import TEST_L0_FILE, VALIDATION_DATA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPECTED_ARRAY_SHAPES = [
    (),  # hi-ialirt  # TODO: Need to implement
    (),  # lo-ialirt  # TODO: Need to implement
    (31778,),  # hskp
    (77, 6, 6, 128),  # lo-counters-aggregated
    (77, 24, 6, 128),  # lo-counters-singles
    (77, 1, 12, 128),  # lo-sw-priority
    (77, 1, 12, 128),  # lo-nsw-priority
    (77, 1, 1, 128),  # lo-sw-species
    (77, 1, 1, 128),  # lo-nsw-species
    (77, 5, 12, 128),  # lo-sw-angular
    (77, 19, 12, 128),  # lo-nsw-angular
    (77, 1, 6, 1),  # hi-counters-aggregated
    (77, 1, 12, 1),  # hi-counters-singles
    (77, 15, 4, 1),  # hi-omni
    (77, 8, 12, 12),  # hi-sectored
    (),  # hi-priority  # TODO: Need to implement
    (),  # lo-pha  # TODO: Need to implement
    (),  # hi-pha  # TODO: Need to implement
]

EXPECTED_LOGICAL_SOURCES = [
    "imap_codice_l1a_hi-ialirt",
    "imap_codice_l1a_lo-ialirt",
    "imap_codice_l1a_hskp",
    "imap_codice_l1a_lo-counters-aggregated",
    "imap_codice_l1a_lo-counters-singles",
    "imap_codice_l1a_lo-sw-priority",
    "imap_codice_l1a_lo-nsw-priority",
    "imap_codice_l1a_lo-sw-species",
    "imap_codice_l1a_lo-nsw-species",
    "imap_codice_l1a_lo-sw-angular",
    "imap_codice_l1a_lo-nsw-angular",
    "imap_codice_l1a_hi-counters-aggregated",
    "imap_codice_l1a_hi-counters-singles",
    "imap_codice_l1a_hi-omni",
    "imap_codice_l1a_hi-sectored",
    "imap_codice_l1a_hi-priority",
    "imap_codice_l1a_lo-pha",
    "imap_codice_l1a_hi-pha",
]

EXPECTED_NUM_VARIABLES = [
    0,  # hi-ialirt  # TODO: Need to implement
    0,  # lo-ialirt  # TODO: Need to implement
    148,  # hskp
    3,  # lo-counters-aggregated
    3,  # lo-counters-singles
    7,  # lo-sw-priority
    4,  # lo-nsw-priority
    18,  # lo-sw-species
    10,  # lo-nsw-species
    6,  # lo-sw-angular
    3,  # lo-nsw-angular
    1,  # hi-counters-aggregated
    3,  # hi-counters-singles
    8,  # hi-omni
    4,  # hi-sectored
    0,  # hi-priority  # TODO: Need to implement
    0,  # lo-pha  # TODO: Need to implement
    0,  # hi-pha  # TODO: Need to implement
]


@pytest.fixture(scope="session")
def test_l1a_data() -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    processed_datasets : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    """

    processed_datasets = process_codice_l1a(file_path=TEST_L0_FILE, data_version="001")

    return processed_datasets


@pytest.mark.parametrize("index", range(len(EXPECTED_ARRAY_SHAPES)))
def test_l1a_data_array_shape(test_l1a_data, index):
    """Tests that the data arrays in the generated CDFs have the expected shape.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    processed_dataset = test_l1a_data[index]
    expected_shape = EXPECTED_ARRAY_SHAPES[index]

    # Mark currently broken/unsupported datasets as expected to fail
    # TODO: Remove these once they are supported
    if index in [0, 1, 15, 16, 17]:
        pytest.xfail("Data product is currently unsupported")

    for variable in processed_dataset:
        if variable in ["energy_table", "acquisition_time_per_step"]:
            assert processed_dataset[variable].data.shape == (128,)
        else:
            assert processed_dataset[variable].data.shape == expected_shape


@pytest.mark.parametrize("index", range(len(EXPECTED_LOGICAL_SOURCES)))
def test_l1a_logical_sources(test_l1a_data, index):
    """Tests that the Logical source of the dataset is what is expected.

    Since the logical source gets set by ``write_cdf``, this also tests that
    the dataset can be written to a file.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    processed_dataset = test_l1a_data[index]
    expected_logical_source = EXPECTED_LOGICAL_SOURCES[index]

    # Mark currently broken/unsupported datasets as expected to fail
    # TODO: Remove these once they are supported
    if index in [0, 1, 2, 15, 16, 17]:
        pytest.xfail("Data product is currently unsupported")

    # Write the dataset to a file to set the logical source attribute
    _ = write_cdf(processed_dataset)

    assert processed_dataset.attrs["Logical_source"] == expected_logical_source


@pytest.mark.parametrize("index", range(len(EXPECTED_NUM_VARIABLES)))
def test_l1a_num_variables(test_l1a_data, index):
    """Tests that the data arrays in the generated CDFs have the expected number
    of variables.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    processed_dataset = test_l1a_data[index]

    # Mark currently broken/unsupported datasets as expected to fail
    # TODO: Remove these once they are supported
    if index in [0, 1, 15, 16, 17]:
        pytest.xfail("Data product is currently unsupported")

    assert len(processed_dataset) == EXPECTED_NUM_VARIABLES[index]


@pytest.mark.skip("Awaiting validation data")
@pytest.mark.parametrize("index", range(len(VALIDATION_DATA)))
def test_l1a_data_array_values(test_l1a_data: xr.Dataset, index):
    """Tests that the generated L1a CDF contents are valid.

    Once proper validation files are acquired, this test function should point
    to those. This function currently just serves as a framework for validating
    files, but does not actually validate them.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    generated_dataset = test_l1a_data
    validation_dataset = load_cdf(VALIDATION_DATA[index])

    # Ensure the processed data matches the validation data
    for variable in validation_dataset:
        assert variable in generated_dataset
        if variable != "epoch":
            np.testing.assert_array_equal(
                validation_dataset[variable].data, generated_dataset[variable].data[0]
            )


def test_l1a_multiple_packets():
    """Tests that an input L0 file containing multiple APIDs can be processed."""

    processed_datasets = process_codice_l1a(file_path=TEST_L0_FILE, data_version="001")

    # TODO: Could add some more checks here?
    assert len(processed_datasets) == 18
