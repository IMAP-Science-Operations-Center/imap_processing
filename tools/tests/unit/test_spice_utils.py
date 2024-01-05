from datetime import datetime
from pathlib import Path

import pytest
import spiceypy as spice

from tools.spice.spice_utils import (
    list_all_constants,
    list_attitude_coverage,
    list_files_with_extensions,
    list_loaded_kernels,
)


@pytest.fixture()
def kernels():
    """Return the SPICE kernels used for testing"""
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    kernels = list_files_with_extensions(directory, [".bsp", ".ti"])
    return kernels


def test_list_files_with_extensions(kernels):
    """Tests the list_files_with_extensions function."""

    directory = Path(__file__).parent.parent / "test_data" / "spice"

    # Test listing files with specified extensions
    result = list_files_with_extensions(directory, [".bsp", ".ti"])
    expected_files = [
        str(directory / "imap_lo_starsensor_instrument_demo.ti"),
        str(directory / "imap_spk_demo.bsp"),
        str(directory / "imap_ultra_instrument_demo.ti"),
    ]
    assert sorted(result) == sorted(expected_files)

    # Test case sensitivity in extensions
    result_case_sensitive = list_files_with_extensions(directory, [".BSP", ".TI"])
    assert result_case_sensitive == expected_files

    # Test with non-matching extensions (should return an empty list)
    result_non_matching = list_files_with_extensions(directory, [".xyz"])
    assert result_non_matching == []


def test_list_loaded_kernels(kernels):
    """Tests the ``list_loaded_kernels`` function"""
    directory = Path(__file__).parent.parent / "test_data" / "spice"

    with spice.KernelPool(kernels):
        result = list_loaded_kernels()

    expected = [
        str(directory / "imap_lo_starsensor_instrument_demo.ti"),
        str(directory / "imap_spk_demo.bsp"),
        str(directory / "imap_ultra_instrument_demo.ti"),
    ]

    assert result == expected


def test_list_all_constants():
    """Tests the ``list_all_constants`` function"""

    # Set up the test environment
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    kernels = list_files_with_extensions(directory, [".tls"])

    with spice.KernelPool(kernels):
        result = list_all_constants()

    # Expected keys
    expected_keys = [
        "DELTET/DELTA_AT",
        "DELTET/DELTA_T_A",
        "DELTET/EB",
        "DELTET/K",
        "DELTET/M",
    ]

    # Assertions
    assert isinstance(result, dict)
    assert list(result.keys()) == expected_keys


def test_list_attitude_coverage():
    """Tests the ``list_attitude_coverage`` function"""

    # Set up the test environment
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    kernels = list_files_with_extensions(directory, [".ah.bc", ".ah.a"])

    with spice.KernelPool(kernels):
        # Test with valid extensions
        result = list_attitude_coverage()

        # Test with invalid custom pattern
        with pytest.raises(ValueError, match=r"Invalid pattern: .*"):
            list_attitude_coverage(r"invalid")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(date, datetime) for date in result)

    # Test with an empty directory
    empty_result = list_attitude_coverage()
    assert empty_result is tuple()
