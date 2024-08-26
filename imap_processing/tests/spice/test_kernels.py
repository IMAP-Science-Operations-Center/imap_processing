"""Tests coverage for imap_processing/spice/kernels.py"""

import numpy as np
import pytest
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError

from imap_processing.spice.kernels import (
    _average_quaternions,
    _create_rotation_matrix,
    _get_et_times,
    create_pointing_frame,
    ensure_spice,
)


@pytest.fixture()
def pointing_frame_kernels(spice_test_data_path):
    """List SPICE kernels."""
    required_kernels = [
        "imap_science_0001.tf",
        "imap_sclk_0000.tsc",
        "imap_sim_ck_2hr_2secsampling_with_nutation.bc",
        "imap_wkcp.tf",
        "naif0012.tls",
    ]
    kernels = [str(spice_test_data_path / kernel) for kernel in required_kernels]
    return kernels


@pytest.fixture()
def et_times(pointing_frame_kernels):
    """Tests get_et_times function."""
    spice.furnsh(pointing_frame_kernels)

    file, _, _, _ = spice.kdata(0, "ck")
    et_start, et_end, et_times = _get_et_times(file)

    return et_times


@ensure_spice
def single_wrap_et2utc(et, fmt, prec):
    """Directly decorate a spice function with ensure_spice for use in tests"""
    return spice.et2utc(et, fmt, prec)


@ensure_spice
def double_wrap_et2utc(et, fmt, prec):
    """Decorate a spice function twice with ensure_spice for use in tests. This
    simulates some decorated outer functions that call lower level functions
    that are already decorated."""
    return single_wrap_et2utc(et, fmt, prec)


@ensure_spice(time_kernels_only=True)
def single_wrap_et2utc_tk_only(et, fmt, prec):
    """Directly wrap a spice function with optional time_kernels_only set True"""
    return spice.et2utc(et, fmt, prec)


@ensure_spice(time_kernels_only=True)
def double_wrap_et2utc_tk_only(et, fmt, prec):
    """Decorate a spice function twice with ensure_spice for use in tests. This
    simulates some decorated outer functions that call lower level functions
    that are already decorated."""
    return single_wrap_et2utc(et, fmt, prec)


@pytest.mark.parametrize(
    "func",
    [
        single_wrap_et2utc,
        single_wrap_et2utc_tk_only,
        double_wrap_et2utc,
        double_wrap_et2utc_tk_only,
    ],
)
def test_ensure_spice_emus_mk_path(func, use_test_metakernel):
    """Test functionality of ensure spice with SPICE_METAKERNEL set"""
    assert func(577365941.184, "ISOC", 3) == "2018-04-18T23:24:31.998"


def test_ensure_spice_time_kernels():
    """Test functionality of ensure spice with timekernels set"""
    wrapped = ensure_spice(spice.et2utc, time_kernels_only=True)
    # TODO: Update/remove this test when a decision has been made about
    #   whether IMAP will use the time_kernels_only functionality and the
    #   ensure_spice decorator has been update.
    with pytest.raises(NotImplementedError):
        _ = wrapped(577365941.184, "ISOC", 3) == "2018-04-18T23:24:31.998"


def test_ensure_spice_key_error():
    """Test functionality of ensure spice when all branches fail"""
    wrapped = ensure_spice(spice.et2utc)
    # The ensure_spice decorator should raise a SpiceyError when all attempts to
    # furnish a set of kernels with sufficient coverage for the spiceypy
    # functions that it decorates.
    with pytest.raises(SpiceyError):
        _ = wrapped(577365941.184, "ISOC", 3) == "2018-04-18T23:24:31.998"


def test_average_quaternions(et_times, pointing_frame_kernels):
    """Tests average_quaternions function."""
    spice.furnsh(pointing_frame_kernels)
    q_avg = _average_quaternions(et_times)

    # Generated from MATLAB code results
    q_avg_expected = np.array([-0.6611, 0.4981, -0.5019, -0.2509])
    np.testing.assert_allclose(q_avg, q_avg_expected, atol=1e-4)


def test_create_rotation_matrix(et_times, pointing_frame_kernels):
    """Tests create_rotation_matrix function."""
    spice.furnsh(pointing_frame_kernels)
    rotation_matrix = _create_rotation_matrix(et_times)
    q_avg = _average_quaternions(et_times)
    z_avg = spice.q2m(list(q_avg))[:, 2]

    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    z_avg_expected = np.array([0.4136, 0.9104, 0.0000])

    np.testing.assert_allclose(z_avg, z_avg_expected, atol=1e-4)
    np.testing.assert_allclose(rotation_matrix, rotation_matrix_expected, atol=1e-4)


def test_create_pointing_frame(spice_test_data_path, pointing_frame_kernels, tmp_path):
    """Tests create_pointing_frame function."""
    spice.furnsh(pointing_frame_kernels)
    ck_kernel, _, _, _ = spice.kdata(0, "ck")
    et_start, et_end, et_times = _get_et_times(ck_kernel)
    create_pointing_frame(pointing_frame_path=tmp_path / "imap_dps.bc")

    # After imap_dps.bc has been created.
    dps_kernel = str(tmp_path / "imap_dps.bc")

    spice.furnsh(dps_kernel)
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
    assert (tmp_path / "imap_dps.bc").exists()


@ensure_spice
def test_et_times(pointing_frame_kernels):
    """Tests get_et_times function."""
    spice.furnsh(pointing_frame_kernels)

    file, _, _, _ = spice.kdata(0, "ck")
    et_start, et_end, et_times = _get_et_times(file)

    assert et_start == 802008069.184905
    assert et_end == 802015267.184906
    assert et_times[0] == et_start
    assert et_times[-1] == et_end

    return et_times
