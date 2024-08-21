"""Tests Pointing Frame Generation."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import spiceypy as spice

from imap_processing.spice.pointing_frame_handler import (
    average_quaternions,
    create_pointing_frame,
    create_rotation_matrix,
    get_et_times,
)


@pytest.fixture()
def kernel_path(tmp_path):
    """Create path to kernels."""

    test_dir = (
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
        / "tests"
        / "spice"
        / "test_data"
    )

    kernels = [
        "naif0012.tls",
        "imap_science_0001.tf",
        "imap_sclk_0000.tsc",
        "imap_wkcp.tf",
        "imap_spin.bc",
    ]

    for file in test_dir.iterdir():
        if file.name in kernels:
            shutil.copy(file, tmp_path / file.name)

    return tmp_path


@pytest.fixture()
def kernels(kernel_path):
    """Create kernel list."""
    kernels = [str(file) for file in kernel_path.iterdir()]

    return kernels


@pytest.fixture()
def ck_kernel(kernel_path):
    """Create ck kernel."""
    ck_kernel = [
        str(file) for file in kernel_path.iterdir() if file.name == "imap_spin.bc"
    ]

    return ck_kernel


@pytest.fixture()
def et_times(ck_kernel, kernels):
    """Tests get_et_times function."""

    with spice.KernelPool(kernels):
        et_start, et_end, et_times = get_et_times(str(ck_kernel[0]))

    return et_times


def test_get_et_times(kernels, ck_kernel):
    """Tests get_et_times function."""

    with spice.KernelPool(kernels):
        et_start, et_end, et_times = get_et_times(str(ck_kernel[0]))

    assert et_start == 802008069.184905
    assert et_end == 802015267.184906
    assert len(et_times) == 57600


def test_average_quaternions(et_times, kernels):
    """Tests average_quaternions function."""

    with spice.KernelPool(kernels):
        q_avg = average_quaternions(et_times)

    # Generated from MATLAB code results
    q_avg_expected = np.array([-0.6838, 0.5480, -0.4469, -0.1802])
    np.testing.assert_allclose(q_avg, q_avg_expected, atol=1e-1)


def test_create_rotation_matrix(et_times, kernels):
    """Tests create_rotation_matrix function."""

    with spice.KernelPool(kernels):
        rotation_matrix = create_rotation_matrix(et_times)
        q_avg = average_quaternions(et_times)
        z_avg = spice.q2m(list(q_avg))[:, 2]

    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    z_avg_expected = np.array([0.4136, 0.9104, 0.0000])

    np.testing.assert_allclose(z_avg, z_avg_expected, atol=1e-4)
    np.testing.assert_allclose(rotation_matrix, rotation_matrix_expected, atol=1e-4)


def test_create_pointing_frame(monkeypatch, kernel_path, ck_kernel):
    """Tests create_pointing_frame function."""
    monkeypatch.setenv("EFS_MOUNT_PATH", str(kernel_path))
    create_pointing_frame()

    # After imap_dps.bc has been created.
    kernels = [str(file) for file in kernel_path.iterdir()]

    with spice.KernelPool(kernels):
        et_start, et_end, et_times = get_et_times(str(ck_kernel[0]))

        rotation_matrix_1 = spice.pxform("ECLIPJ2000", "IMAP_DPS", et_start + 100)
        rotation_matrix_2 = spice.pxform("ECLIPJ2000", "IMAP_DPS", et_start + 1000)

    # All the rotation matrices should be the same.
    assert np.array_equal(rotation_matrix_1, rotation_matrix_2)

    # Nick Dutton's MATLAB code result
    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    np.testing.assert_allclose(rotation_matrix_1, rotation_matrix_expected, atol=1e-4)

    # Verify imap_dps.bc has been created.
    assert (kernel_path / "imap_dps.bc").exists()
