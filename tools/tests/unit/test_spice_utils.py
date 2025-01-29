from pathlib import Path

import pytest
import spiceypy

from tools.spice.spice_utils import (
    list_all_constants,
    list_files_with_extensions,
    list_loaded_kernels,
)


@pytest.fixture()
def kernels():
    """Return the SPICE kernels used for testing"""
    # TODO: ALl kernels able to be downloaded from NAIF are not available
    #  in the test_data/spice directory.
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    kernels = list_files_with_extensions(directory, [".bsp", ".tf"])
    return kernels


@pytest.mark.xfail(reason="Download NAIF kernels")
def test_list_files_with_extensions(kernels):
    """Tests the list_files_with_extensions function."""

    directory = Path(__file__).parent.parent / "test_data" / "spice"

    # Test listing files with specified extensions
    result = list_files_with_extensions(directory, [".tf"])
    expected_files = [
        str(directory / "imap_wkcp.tf"),
    ]
    assert sorted(result) == sorted(expected_files)

    # Test case sensitivity in extensions
    result_case_sensitive = list_files_with_extensions(directory, [".TF"])
    assert sorted(result_case_sensitive) == sorted(expected_files)

    # Test with non-matching extensions (should return an empty list)
    result_non_matching = list_files_with_extensions(directory, [".xyz"])
    assert result_non_matching == []


@pytest.mark.xfail(reason="Download NAIF kernels")
def test_list_loaded_kernels(kernels):
    """Tests the ``list_loaded_kernels`` function"""
    directory = Path(__file__).parent.parent / "test_data" / "spice"

    with spiceypy.KernelPool(kernels):
        result = list_loaded_kernels()

    expected = [
        str(directory / "imap_wkcp.tf"),
        str(directory / "IMAP_launch20250429_1D.bsp"),
        str(directory / "L1_de431.bsp"),
        str(directory / "de430.bsp"),
    ]

    assert sorted(result) == sorted(expected)


@pytest.mark.xfail(reason="Download NAIF kernels")
def test_list_all_constants():
    """Tests the ``list_all_constants`` function"""

    # Set up the test environment
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    kernels = list_files_with_extensions(directory, [".tsc"])

    with spiceypy.KernelPool(kernels):
        result = list_all_constants()

    # Expected keys
    expected_keys = [
        "SCLK01_COEFFICIENTS_43",
        "SCLK01_MODULI_43",
        "SCLK01_N_FIELDS_43",
        "SCLK01_OFFSETS_43",
        "SCLK01_OUTPUT_DELIM_43",
        "SCLK01_TIME_SYSTEM_43",
        "SCLK_DATA_TYPE_43",
        "SCLK_DATA_TYPE_43000",
        "SCLK_KERNEL_ID",
        "SCLK_PARTITION_END_43",
        "SCLK_PARTITION_START_43",
    ]

    # Assertions
    assert isinstance(result, dict)
    assert list(result.keys()) == expected_keys
