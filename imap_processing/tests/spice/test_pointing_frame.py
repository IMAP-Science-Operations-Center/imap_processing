"""Test coverage for imap_processing.spice.repoint.py"""

import numpy as np
import pytest
import spiceypy

from imap_processing.spice.pointing_frame import (
    _average_quaternions,
    _create_rotation_matrix,
    _get_et_times,
    create_pointing_frame,
)

from imap_processing.spice.repoint import get_repoint_data


@pytest.fixture
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


@pytest.fixture
def et_times(pointing_frame_kernels):
    """Tests get_et_times function."""
    spiceypy.furnsh(pointing_frame_kernels)

    ck_kernel, _, _, _ = spiceypy.kdata(0, "ck")
    ck_cover = spiceypy.ckcov(ck_kernel, -43000, True, "INTERVAL", 0, "TDB")
    et_start, et_end = spiceypy.wnfetd(ck_cover, 0)
    et_times = _get_et_times(et_start, et_end)

    return et_times


@pytest.fixture
def fake_repoint_data(monkeypatch, spice_test_data_path):
    """Generate fake spin dataframe for testing"""
    fake_repoint_path = spice_test_data_path / "fake_repoint_data.csv"
    monkeypatch.setenv("REPOINT_DATA_FILEPATH", str(fake_repoint_path))
    return fake_repoint_path


def test_average_quaternions(et_times, pointing_frame_kernels):
    """Tests average_quaternions function."""
    spiceypy.furnsh(pointing_frame_kernels)
    q_avg = _average_quaternions(et_times)

    # Generated from MATLAB code results
    q_avg_expected = np.array([-0.6611, 0.4981, -0.5019, -0.2509])
    np.testing.assert_allclose(q_avg, q_avg_expected, atol=1e-4)


def test_create_rotation_matrix(et_times, pointing_frame_kernels):
    """Tests create_rotation_matrix function."""
    spiceypy.furnsh(pointing_frame_kernels)
    rotation_matrix = _create_rotation_matrix(et_times)
    q_avg = _average_quaternions(et_times)
    z_avg = spiceypy.q2m(list(q_avg))[:, 2]

    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    z_avg_expected = np.array([0.4136, 0.9104, 0.0000])

    np.testing.assert_allclose(z_avg, z_avg_expected, atol=1e-4)
    np.testing.assert_allclose(rotation_matrix, rotation_matrix_expected, atol=1e-4)


def test_create_pointing_frame(
    spice_test_data_path, pointing_frame_kernels, tmp_path, et_times
):
    """Tests create_pointing_frame function."""
    spiceypy.kclear()
    spiceypy.furnsh(pointing_frame_kernels)
    create_pointing_frame(
        pointing_frame_path=tmp_path / "imap_dps.bc",
        ck_path=spice_test_data_path / "imap_sim_ck_2hr_2secsampling_with_nutation.bc",
    )

    # After imap_dps.bc has been created.
    dps_kernel = str(tmp_path / "imap_dps.bc")

    spiceypy.furnsh(dps_kernel)
    rotation_matrix_1 = spiceypy.pxform("ECLIPJ2000", "IMAP_DPS", et_times[0] + 100)
    rotation_matrix_2 = spiceypy.pxform("ECLIPJ2000", "IMAP_DPS", et_times[0] + 1000)

    # All the rotation matrices should be the same.
    assert np.array_equal(rotation_matrix_1, rotation_matrix_2)

    # Nick Dutton's MATLAB code result
    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    np.testing.assert_allclose(rotation_matrix_1, rotation_matrix_expected, atol=1e-4)

    # Verify imap_dps.bc has been created.
    assert (tmp_path / "imap_dps.bc").exists()

    # Tests error handling when incorrect kernel is loaded.
    spiceypy.furnsh(pointing_frame_kernels)
    with pytest.raises(
        ValueError, match="Error: Expected CK kernel badname_kernel.bc"
    ):  # Replace match string with expected error message
        create_pointing_frame(
            pointing_frame_path=tmp_path / "imap_dps.bc", ck_path="badname_kernel.bc"
        )


def test_et_times(pointing_frame_kernels):
    """Tests get_et_times function."""
    spiceypy.furnsh(pointing_frame_kernels)

    ck_kernel, _, _, _ = spiceypy.kdata(0, "ck")
    ck_cover = spiceypy.ckcov(ck_kernel, -43000, True, "INTERVAL", 0, "TDB")
    et_start, et_end = spiceypy.wnfetd(ck_cover, 0)
    et_times = _get_et_times(et_start, et_end)

    assert et_times[0] == et_start
    assert et_times[-1] == et_end

    return et_times


def test_multiple_attempts(pointing_frame_kernels, tmp_path, spice_test_data_path):
    """Tests create_pointing_frame function with multiple pointing kernels."""
    spiceypy.furnsh(pointing_frame_kernels)

    # Check that a single segment is added regardless of how many times
    # create_pointing_frame is called.
    create_pointing_frame(
        pointing_frame_path=tmp_path / "imap_dps.bc",
        ck_path=spice_test_data_path / "imap_sim_ck_2hr_2secsampling_with_nutation.bc",
    )
    ck_cover = spiceypy.ckcov(
        str(tmp_path / "imap_dps.bc"), -43901, True, "INTERVAL", 0, "TDB"
    )
    num_intervals = spiceypy.wncard(ck_cover)
    assert num_intervals == 1

    create_pointing_frame(
        pointing_frame_path=tmp_path / "imap_dps.bc",
        ck_path=spice_test_data_path / "imap_sim_ck_2hr_2secsampling_with_nutation.bc",
    )
    ck_cover = spiceypy.ckcov(
        str(tmp_path / "imap_dps.bc"), -43901, True, "INTERVAL", 0, "TDB"
    )
    num_intervals = spiceypy.wncard(ck_cover)
    assert num_intervals == 1


def test_multiple_pointings(pointing_frame_kernels, spice_test_data_path, tmp_path):
    """Tests create_pointing_frame function with multiple pointing kernels."""
    spiceypy.furnsh(pointing_frame_kernels)

    create_pointing_frame(
        pointing_frame_path=tmp_path / "imap_pointing_frame.bc",
        ck_path=spice_test_data_path / "imap_sim_ck_2hr_2secsampling_with_nutation.bc",
    )
    ck_cover_pointing = spiceypy.ckcov(
        str(tmp_path / "imap_pointing_frame.bc"),
        -43901,
        True,
        "INTERVAL",
        0,
        "TDB",
    )
    num_intervals = spiceypy.wncard(ck_cover_pointing)
    et_start_pointing, et_end_pointing = spiceypy.wnfetd(ck_cover_pointing, 0)

    ck_cover = spiceypy.ckcov(
        str(spice_test_data_path / "imap_sim_ck_2hr_2secsampling_with_nutation.bc"),
        -43000,
        True,
        "INTERVAL",
        0,
        "TDB",
    )
    num_intervals_expected = spiceypy.wncard(ck_cover)
    et_start_expected, et_end_expected = spiceypy.wnfetd(ck_cover, 0)

    assert num_intervals == num_intervals_expected
    assert et_start_pointing == et_start_expected
    assert et_end_pointing == et_end_expected

    et_times = _get_et_times(et_start_pointing, et_end_pointing)

    spiceypy.furnsh(str(tmp_path / "imap_pointing_frame.bc"))
    rotation_matrix_1 = spiceypy.pxform("ECLIPJ2000", "IMAP_DPS", et_times[100])
    rotation_matrix_2 = spiceypy.pxform("ECLIPJ2000", "IMAP_DPS", et_times[1000])

    assert np.array_equal(rotation_matrix_1, rotation_matrix_2)

