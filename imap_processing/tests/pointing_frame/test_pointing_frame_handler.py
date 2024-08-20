"""Tests Pointing Frame Generation."""

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import spiceypy as spice

from imap_processing.pointing_frame_handler import (
    average_quaternions,
    create_pointing_frame,
    create_rotation_matrix,
    get_coverage,
)


@pytest.fixture()
def kernel_path(tmp_path):
    """Create path to kernels."""

    test_dir = (
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
        / "tests"
        / "pointing_frame"
        / "test_data"
    )

    kernels = [
        "de430.bsp",
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
def create_kernel_list(kernel_path):
    """Create kernel lists."""
    kernels = [str(file) for file in kernel_path.iterdir()]
    ck_kernel = [
        str(file) for file in kernel_path.iterdir() if file.name == "imap_spin.bc"
    ]

    return kernels, ck_kernel


@pytest.fixture()
def et_times(create_kernel_list):
    """Tests get_coverage function."""
    kernels, ck_kernel = create_kernel_list

    with spice.KernelPool(kernels):
        et_start, et_end, et_times = get_coverage(str(ck_kernel[0]))

    return et_times


@pytest.mark.xfail(reason="Will fail unless kernels in pointing_frame/test_data.")
def test_get_coverage(create_kernel_list):
    """Tests get_coverage function."""
    kernels, ck_kernel = create_kernel_list

    with spice.KernelPool(kernels):
        et_start, et_end, et_times = get_coverage(str(ck_kernel[0]))

    assert et_start == 802008069.184905
    assert et_end == 802094467.184905


@pytest.mark.xfail(reason="Will fail unless kernels in pointing_frame/test_data.")
def test_average_quaternions(et_times, create_kernel_list):
    """Tests average_quaternions function."""

    kernels, ck_kernel = create_kernel_list
    with spice.KernelPool(kernels):
        q_avg, z_eclip_time = average_quaternions(et_times)

    # Generated from MATLAB code results
    q_avg_expected = np.array([-0.6838, 0.5480, -0.4469, -0.1802])
    np.testing.assert_allclose(q_avg, q_avg_expected, atol=1e-4)


@pytest.mark.xfail(reason="Will fail unless kernels in pointing_frame/test_data.")
def test_create_rotation_matrix(et_times, kernel_path):
    """Tests create_rotation_matrix function."""

    kernels = [str(file) for file in kernel_path.iterdir()]

    with spice.KernelPool(kernels):
        rotation_matrix, z_avg = create_rotation_matrix(et_times)

    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    z_avg_expected = np.array([0.4136, 0.9104, 0.0000])

    np.testing.assert_allclose(z_avg, z_avg_expected, atol=1e-4)
    np.testing.assert_allclose(rotation_matrix, rotation_matrix_expected, atol=1e-4)


@pytest.mark.xfail(reason="Will fail unless kernels in pointing_frame/test_data.")
def test_create_pointing_frame(monkeypatch, kernel_path, create_kernel_list):
    """Tests create_pointing_frame function."""
    monkeypatch.setenv("EFS_MOUNT_PATH", str(kernel_path))
    _, ck_kernel = create_kernel_list
    create_pointing_frame()

    # After imap_dps.bc has been created.
    kernels = [str(file) for file in kernel_path.iterdir()]

    with spice.KernelPool(kernels):
        et_start, et_end, et_times = get_coverage(str(ck_kernel[0]))

        rotation_matrix_1 = spice.pxform("ECLIPJ2000", "IMAP_DPS", et_start + 100)
        rotation_matrix_2 = spice.pxform("ECLIPJ2000", "IMAP_DPS", et_start + 1000)

    # All the rotation matrices should be the same.
    assert np.array_equal(rotation_matrix_1, rotation_matrix_2)

    # Nick Dutton's MATLAB code result
    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    np.testing.assert_allclose(rotation_matrix_1, rotation_matrix_expected, atol=1e-4)


@pytest.mark.xfail(reason="Will fail unless kernels in pointing_frame/test_data.")
def test_z_axis(create_kernel_list):
    """Tests Inertial z axis and provides visualization."""
    kernels, ck_kernel = create_kernel_list

    with spice.KernelPool(kernels):
        et_start, et_end, et_times = get_coverage(str(ck_kernel[0]))

        # Converts rectangular coordinates to spherical coordinates.
        q_avg, z_eclip_time = average_quaternions(et_times)
        z_avg_expected = spice.q2m(list(q_avg))[:, 2]
        _, z_avg = create_rotation_matrix(et_times)

        assert np.array_equal(z_avg, z_avg_expected)

        # Create visualization
        declination_list = []
        for time in z_eclip_time:
            _, _, declination = spice.recrad(list(time))
            declination_list.append(declination)

        # Average declination.
        _, _, avg_declination = spice.recrad(list(z_avg))

    # Plotting for visualization
    plt.figure()
    plt.plot(
        et_times, np.array(declination_list) * 180 / np.pi, "-b", label="Declination"
    )
    plt.plot(
        et_times,
        np.full(len(et_times), avg_declination * 180 / np.pi),
        "-r",
        linewidth=2,
        label="mean z-axis for pointing frame",
    )
    plt.xlabel("Ephemeris Time")
    plt.ylabel("Spacecraft Spin Axis Declination")
    plt.ticklabel_format(useOffset=False)
    plt.legend()
    plt.show()
