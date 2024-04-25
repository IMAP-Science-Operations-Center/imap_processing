"""Tests the L1a processing for decommutated CoDICE data"""

from pathlib import Path

import cdflib
import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.codice_l1a import process_codice_l1a

EXPECTED_ARRAY_SHAPES = [
    (99,),  # hskp
    (1, 128),  # lo-sw-species-counts
    (1, 112),  # lo-nsw-species-counts
]
EXPECTED_ARRAY_SIZES = [
    123,  # hskp
    16,  # lo-sw-species-counts
    8,  # lo-nsw-species-counts
]
EXPECTED_FILENAMES = [
    "imap_codice_l1a_hskp_20100101_v001.cdf",
    "imap_codice_l1a_lo-sw-species-counts_20240319_v001.cdf",
    "imap_codice_l1a_lo-nsw-species-counts_20240319_v001.cdf",
]
TEST_PACKETS = [
    Path(
        f"{imap_module_directory}/tests/codice/data/raw_ccsds_20230822_122700Z_idle.bin"
    ),
    Path(f"{imap_module_directory}/tests/codice/data/lo_fsw_view_5_ccsds.bin"),
    Path(f"{imap_module_directory}/tests/codice/data/lo_fsw_view_6_ccsds.bin"),
]

# Placeholder for validation data files
VALIDATION_DATA = [
    f"{imap_module_directory}/tests/codice/data/validation_hskp.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-sw-species-counts.cdf",
    f"{imap_module_directory}/tests/codice/data/validation_lo-nsw-species-counts.cdf",
]


@pytest.fixture(params=TEST_PACKETS)
def test_l1a_data(request) -> tuple[xr.Dataset, str]:
    packets = decom_packets(request.param)
    dataset, cdf_filename = process_codice_l1a(packets)
    return dataset, cdf_filename


@pytest.mark.parametrize(
    "test_l1a_data, expected_filename",
    list(zip(TEST_PACKETS, EXPECTED_FILENAMES)),
    indirect=["test_l1a_data"],
)
def test_l1a_cdf_filenames(test_l1a_data, expected_filename: str):
    """Tests that the ``process_codice_l1a`` function generates CDF files with
    expected filenames.

    Parameters
    ----------
    test_l1a_data : tuple
        A tuple containing the ``xarray`` dataset and the CDF filename
    expected_filename : str
        The expected CDF filename
    """

    _, cdf_filename = test_l1a_data
    assert cdf_filename.name == expected_filename


@pytest.mark.parametrize(
    "test_l1a_data, expected_shape",
    list(zip(TEST_PACKETS, EXPECTED_ARRAY_SHAPES)),
    indirect=["test_l1a_data"],
)
def test_l1a_data_array_shape(test_l1a_data, expected_shape: tuple):
    """Tests that the data arrays in the generated CDFs have the expected shape.

    Parameters
    ----------
    test_l1a_data : tuple
        A tuple containing the ``xarray`` dataset and the CDF filename
    expected_shape : tuple
        The expected shape of the data array
    """

    dataset, _ = test_l1a_data
    for variable in dataset:
        assert dataset[variable].data.shape == expected_shape


@pytest.mark.parametrize(
    "test_l1a_data, expected_size",
    list(zip(TEST_PACKETS, EXPECTED_ARRAY_SIZES)),
    indirect=["test_l1a_data"],
)
def test_l1a_data_array_size(test_l1a_data, expected_size: int):
    """Tests that the data arrays in the generated CDFs have the expected size.

    Parameters
    ----------
    test_l1a_data : tuple
        A tuple containing the ``xarray`` dataset and the CDF filename
    expected_size : int
        The expected size of the data array
    """

    dataset, _ = test_l1a_data
    assert len(dataset) == expected_size


@pytest.mark.skip("Awaiting validation data")
@pytest.mark.parametrize(
    "test_l1a_data, validation_data",
    list(zip(TEST_PACKETS, VALIDATION_DATA)),
    indirect=["test_l1a_data"],
)
def test_l1a_data_array_values(test_l1a_data, validation_data: Path):
    """Tests that the generated L1a CDF contents are valid.

    Once proper validation files are acquired, this test function should point
    to those. This function currently just serves as a framework for validating
    files, but does not actually validate them.

    Parameters
    ----------
    test_l1a_data : tuple
        A tuple containing the ``xarray`` dataset and the CDF filename
    validataion_data : Path
        The path to the file containing the validation data
    """

    generated_dataset, _ = test_l1a_data
    validation_dataset = cdflib.xarray.cdf_to_xarray(validation_data)

    # Ensure the processed data matches the validation data
    for variable in validation_dataset:
        assert variable in generated_dataset
        if variable != "epoch":
            np.testing.assert_array_equal(
                validation_data[variable].data, generated_dataset[variable].data[0]
            )
