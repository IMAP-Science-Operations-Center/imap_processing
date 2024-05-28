"""Tests the L1a processing for decommutated CoDICE data"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.codice_l1a import process_codice_l1a

EXPECTED_ARRAY_SHAPES = [
    (99,),  # hskp
    (1, 128),  # lo-sw-angular-counts
    (1, 128),  # lo-nsw-angular-counts
    (1, 128),  # lo-sw-priority-counts
    (1, 128),  # lo-nsw-priority-counts
    (1, 128),  # lo-sw-species-counts
    (1, 128),  # lo-nsw-species-counts
]
EXPECTED_ARRAY_SIZES = [
    123,  # hskp
    6,  # lo-sw-angular-counts
    3,  # lo-nsw-angular-counts
    7,  # lo-sw-priority-counts
    4,  # lo-nsw-priority-counts
    18,  # lo-sw-species-counts
    10,  # lo-nsw-species-counts
]
EXPECTED_FILENAMES = [
    "imap_codice_l1a_hskp_20100101_v001.cdf",
    "imap_codice_l1a_lo-sw-angular-counts_20240429_v001.cdf",
    "imap_codice_l1a_lo-nsw-angular-counts_20240429_v001.cdf",
    "imap_codice_l1a_lo-sw-priority-counts_20240429_v001.cdf",
    "imap_codice_l1a_lo-nsw-priority-counts_20240429_v001.cdf",
    "imap_codice_l1a_lo-sw-species-counts_20240429_v001.cdf",
    "imap_codice_l1a_lo-nsw-species-counts_20240429_v001.cdf",
]
TEST_PACKETS = [
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_hskp_20230822.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-sw-angular_20240429.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-nsw-angular_20240429.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-sw-priority_20240429.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-nsw-priority_20240429.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-sw-species_20240429.pkts"
    ),
    Path(
        f"{imap_module_directory}/tests/codice/data/imap_codice_l0_lo-nsw-species_20240429.pkts"
    ),
]

# Placeholder for validation data files
VALIDATION_DATA = [
    f"{imap_module_directory}/tests/codice/data/validation_hskp.cdf",
    f"{imap_module_directory}/tests/codice/data/validataion_lo-sw-angular-counts.cdf",
    f"{imap_module_directory}/tests/codice/data/validataion_lo-nsw-angular-counts.cdf",
    f"{imap_module_directory}/tests/codice/data/validataion_lo-sw-priority-counts.cdf",
    f"{imap_module_directory}/tests/codice/data/validataion_lo-nsw-priority-counts.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-sw-species-counts.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-nsw-species-counts.cdf",
]


@pytest.fixture(params=TEST_PACKETS)
def test_l1a_data(request) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    packets = decom_packets(request.param)
    dataset = process_codice_l1a(packets)
    return dataset


@pytest.mark.parametrize(
    "test_l1a_data, expected_filename",
    list(zip(TEST_PACKETS, EXPECTED_FILENAMES)),
    indirect=["test_l1a_data"],
)
def test_l1a_cdf_filenames(test_l1a_data: xr.Dataset, expected_filename: str):
    """Tests that the ``process_codice_l1a`` function generates CDF files with
    expected filenames.

    Parameters
    ----------
    test_l1a_data : xr.Dataset
        A ``xarray`` dataset containing the test data
    expected_filename : str
        The expected CDF filename
    """

    dataset = test_l1a_data
    assert dataset.cdf_filename.name == expected_filename


@pytest.mark.parametrize(
    "test_l1a_data, expected_shape",
    list(zip(TEST_PACKETS, EXPECTED_ARRAY_SHAPES)),
    indirect=["test_l1a_data"],
)
def test_l1a_data_array_shape(test_l1a_data: xr.Dataset, expected_shape: tuple):
    """Tests that the data arrays in the generated CDFs have the expected shape.

    Parameters
    ----------
    test_l1a_data : xr.Dataset
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
    test_l1a_data : xr.Dataset
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
    test_l1a_data : xr.Dataset
        A ``xarray`` dataset containing the test data
    validataion_data : Path
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
