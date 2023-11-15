import logging
import os
from datetime import datetime
from pathlib import Path

import pytest
import spiceypy as spice

from tools.spice.spice_utils import (
    SpiceKernelManager,
    ls_attitude_coverage,
    ls_kernels,
    ls_spice_constants,
)


@pytest.fixture()
def kernel_object():
    """Data for decom"""
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    extensions = [".bpc", ".bsp", ".ti", ".tf", ".tls", ".tsc"]
    kernels = SpiceKernelManager(directory, extensions)
    return kernels


def test_furnsh(kernel_object):
    kernel_object.furnsh()
    n_files_loaded = spice.ktotal("ALL")

    extensions = [".bpc", ".bsp", ".ti", ".tf", ".tls", ".tsc"]

    files = [
        file
        for file in os.listdir(kernel_object.kernel_path)
        if file.endswith(tuple(extensions))
    ]

    n_test_files = len(files)

    assert n_files_loaded == n_test_files


def test_clear(kernel_object):
    # Ensure some kernels are loaded first
    kernel_object.furnsh()
    n_files_before_clear = spice.ktotal("ALL")
    assert n_files_before_clear > 0

    # Clear loaded kernels and verify
    kernel_object.clear()
    n_files_loaded_after_clear = spice.ktotal("ALL")
    assert n_files_loaded_after_clear == 0


def test_furnsh_type(caplog):
    caplog.set_level(logging.DEBUG)
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    kernel_object = SpiceKernelManager(directory, [".dog"])

    kernel_object.furnsh()

    expected_error_msg = "Invalid kernel_type extension:"
    assert any(expected_error_msg in message for message in caplog.messages)

    kernel_object.clear()


def test_ls_kernels(kernel_object):
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    extensions = [".bsp"]
    kernel_object = SpiceKernelManager(directory, extensions)

    kernel_object.furnsh()
    result = ls_kernels()

    expected = [str(directory / "imap_spk_demo.bsp")]

    assert sorted(result) == sorted(expected)


def test_ls_spice_constants(kernel_object):
    # Set up the test environment
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    extensions = [".tls"]
    kernel_object = SpiceKernelManager(directory, extensions)
    kernel_object.furnsh()

    result = ls_spice_constants()

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


def test_ls_attitude_coverage(kernel_object):
    # Set up the test environment
    directory = Path(__file__).parent.parent / "test_data" / "spice"
    extensions = [".ah.bc", ".ah.a"]
    kernel_object = SpiceKernelManager(directory, extensions)
    kernel_object.furnsh()

    # Test with valid extensions
    result = ls_attitude_coverage()
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(date, datetime) for date in result)

    # Test with invalid custom pattern
    with pytest.raises(ValueError, match=r"Invalid pattern: .*"):
        ls_attitude_coverage(r"invalid")

    # Test with an empty directory
    kernel_object.clear()
    empty_result = ls_attitude_coverage()
    assert empty_result is None
