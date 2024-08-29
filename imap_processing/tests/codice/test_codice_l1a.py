"""Tests the L1a processing for decommutated CoDICE data"""

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice.codice_l1a import process_codice_l1a

from .conftest import TEST_PACKETS, VALIDATION_DATA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Add test that processes a file with multiple APIDs

EXPECTED_ARRAY_SHAPES = [
    (99,),  # hskp
    (1, 128),  # hi-counters-aggregated
    (1, 128),  # hi-counters-singles
    (1, 128),  # hi-omni
    (1, 128),  # hi-sectored
    (1, 128),  # hi-pha
    (1, 128),  # lo-counters-aggregated
    (1, 128),  # lo-counters-aggregated
    (1, 128),  # lo-sw-angular
    (1, 128),  # lo-nsw-angular
    (1, 128),  # lo-sw-priority
    (1, 128),  # lo-nsw-priority
    (1, 128),  # lo-sw-species
    (1, 128),  # lo-nsw-species
    (1, 128),  # lo-pha
]
EXPECTED_ARRAY_SIZES = [
    123,  # hskp
    1,  # hi-counters-aggregated
    3,  # hi-counters-singles
    8,  # hi-omni
    4,  # hi-sectored
    0,  # hi-pha
    3,  # lo-counters-aggregated
    3,  # lo-counters-singles
    6,  # lo-sw-angular
    3,  # lo-nsw-angular
    7,  # lo-sw-priority
    4,  # lo-nsw-priority
    18,  # lo-sw-species
    10,  # lo-nsw-species
    0,  # lo-pha
]
EXPECTED_LOGICAL_SOURCE = [
    "imap_codice_l1a_hskp",
    "imap_codice_l1a_hi-counters-aggregated",
    "imap_codice_l1a_hi-counters-singles",
    "imap_codice_l1a_hi-omni",
    "imap_codice_l1a_hi-sectored",
    "imap_codice_l1a_hi-pha",
    "imap_codice_l1a_lo-counters-aggregated",
    "imap_codice_l1a_lo-counters-singles",
    "imap_codice_l1a_lo-sw-angular",
    "imap_codice_l1a_lo-nsw-angular",
    "imap_codice_l1a_lo-sw-priority",
    "imap_codice_l1a_lo-nsw-priority",
    "imap_codice_l1a_lo-sw-species",
    "imap_codice_l1a_lo-nsw-species",
    "imap_codice_l1a_lo-pha",
]


@pytest.fixture(params=TEST_PACKETS)
def test_l1a_data(request) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xarray.Dataset
        A ``xarray`` dataset containing the test data
    """

    dataset = process_codice_l1a(file_path=request.param, data_version="001")

    # Write the dataset to a CDF so it can be manually inspected as well
    file_path = write_cdf(dataset)
    logger.info(f"CDF file written to {file_path}")

    return dataset


@pytest.mark.parametrize(
    "test_l1a_data, expected_logical_source",
    list(zip(TEST_PACKETS, EXPECTED_LOGICAL_SOURCE)),
    indirect=["test_l1a_data"],
)
def test_l1a_cdf_filenames(test_l1a_data: xr.Dataset, expected_logical_source: str):
    """Tests that the ``process_codice_l1a`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    test_l1a_data : xarray.Dataset
        A ``xarray`` dataset containing the test data
    expected_logical_source : str
        The expected CDF filename
    """

    dataset = test_l1a_data
    assert dataset.attrs["Logical_source"] == expected_logical_source


@pytest.mark.xfail(
    reason="Currently failing due to cdflib/epoch issue. See https://github.com/MAVENSDC/cdflib/issues/268"
)
@pytest.mark.parametrize(
    "test_l1a_data, expected_shape",
    list(zip(TEST_PACKETS, EXPECTED_ARRAY_SHAPES)),
    indirect=["test_l1a_data"],
)
def test_l1a_data_array_shape(test_l1a_data: xr.Dataset, expected_shape: tuple):
    """Tests that the data arrays in the generated CDFs have the expected shape.

    Parameters
    ----------
    test_l1a_data : xarray.Dataset
        A ``xarray`` dataset containing the test data
    expected_shape : tuple
        The expected shape of the data array
    """

    dataset = test_l1a_data
    for variable in dataset:
        if variable in ["esa_sweep_values", "acquisition_times"]:
            assert dataset[variable].data.shape == (128,)
        else:
            assert dataset[variable].data.shape == expected_shape


@pytest.mark.parametrize(
    "test_l1a_data, expected_size",
    list(zip(TEST_PACKETS, EXPECTED_ARRAY_SIZES)),
    indirect=["test_l1a_data"],
)
def test_l1a_data_array_size(test_l1a_data: xr.Dataset, expected_size: int):
    """Tests that the data arrays in the generated CDFs have the expected size.

    Parameters
    ----------
    test_l1a_data : xarray.Dataset
        A ``xarray`` dataset containing the test data
    expected_size : int
        The expected size of the data array
    """

    dataset = test_l1a_data
    assert len(dataset) == expected_size


@pytest.mark.skip("Awaiting validation data")
@pytest.mark.parametrize(
    "test_l1a_data, validation_data",
    list(zip(TEST_PACKETS, VALIDATION_DATA)),
    indirect=["test_l1a_data"],
)
def test_l1a_data_array_values(test_l1a_data: xr.Dataset, validation_data: Path):
    """Tests that the generated L1a CDF contents are valid.

    Once proper validation files are acquired, this test function should point
    to those. This function currently just serves as a framework for validating
    files, but does not actually validate them.

    Parameters
    ----------
    test_l1a_data : xarray.Dataset
        A ``xarray`` dataset containing the test data
    validataion_data : pathlib.Path
        The path to the file containing the validation data
    """

    generated_dataset = test_l1a_data
    validation_dataset = load_cdf(validation_data)

    # Ensure the processed data matches the validation data
    for variable in validation_dataset:
        assert variable in generated_dataset
        if variable != "epoch":
            np.testing.assert_array_equal(
                validation_data[variable].data, generated_dataset[variable].data[0]
            )
