"""Tests the L1a processing for decommutated CoDICE data"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l1a import process_codice_l1a

# TODO: Add test that processes a file with multiple APIDs

EXPECTED_ARRAY_SHAPES = [
    (99,),  # hskp
    (1, 128),  # hi-counters-aggregated
    (1, 128),  # hi-counters-singles
    (1, 128),  # hi-omni
    (1, 128),  # hi-sectored
    (1, 128),  # lo-counters-aggregated
    (1, 128),  # lo-counters-aggregated
    (1, 128),  # lo-sw-angular
    (1, 128),  # lo-nsw-angular
    (1, 128),  # lo-sw-priority
    (1, 128),  # lo-nsw-priority
    (1, 128),  # lo-sw-species
    (1, 128),  # lo-nsw-species
]
EXPECTED_ARRAY_SIZES = [
    123,  # hskp
    1,  # hi-counters-aggregated
    3,  # hi-counters-singles
    8,  # hi-omni
    4,  # hi-sectored
    3,  # lo-counters-aggregated
    3,  # lo-counters-singles
    6,  # lo-sw-angular
    3,  # lo-nsw-angular
    7,  # lo-sw-priority
    4,  # lo-nsw-priority
    18,  # lo-sw-species
    10,  # lo-nsw-species
]
EXPECTED_LOGICAL_SOURCE = [
    "imap_codice_l1a_hskp",
    "imap_codice_l1a_hi-counters-aggregated",
    "imap_codice_l1a_hi-counters-singles",
    "imap_codice_l1a_hi-omni",
    "imap_codice_l1a_hi-sectored",
    "imap_codice_l1a_lo-counters-aggregated",
    "imap_codice_l1a_lo-counters-singles",
    "imap_codice_l1a_lo-sw-angular",
    "imap_codice_l1a_lo-nsw-angular",
    "imap_codice_l1a_lo-sw-priority",
    "imap_codice_l1a_lo-nsw-priority",
    "imap_codice_l1a_lo-sw-species",
    "imap_codice_l1a_lo-nsw-species",
]
TEST_PACKETS = [
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_hskp_20100101_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_hi-counters-aggregated_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_hi-counters-singles_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_hi-omni_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_hi-sectored_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-counters-aggregated_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-counters-singles_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-sw-angular_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-nsw-angular_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-sw-priority_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-nsw-priority_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-sw-species_20240429_v001.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-nsw-species_20240429_v001.pkts"
    ),
]

# Placeholder for validation data files
VALIDATION_DATA = [
    f"{imap_module_directory}/tests/codice/data/validation_hskp.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_hi-counters-aggregated.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_hi-counters-singles.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_hi-omni.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_hi-sectored.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-counters-aggregated.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-counters-singles.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-sw-angular.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-nsw-angular.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-sw-priority.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-nsw-priority.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-sw-species.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-nsw-species.cdf",
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
    reason="Currently failing due to cdflib/epoch issue. Revisit after SIT-3"
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
