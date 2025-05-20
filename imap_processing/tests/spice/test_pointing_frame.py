"""Test coverage for imap_processing.spice.repoint.py"""

import numpy as np
import pytest
import spiceypy

from imap_processing.spice import IMAP_SC_ID
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.pointing_frame import (
    POINTING_SEGMENT_DTYPE,
    _average_quaternions,
    _create_rotation_matrix,
    calculate_pointing_attitude_segments,
    write_pointing_frame_ck,
)
from imap_processing.spice.time import TICK_DURATION


@pytest.fixture
def pointing_frame_kernels(spice_test_data_path):
    """List SPICE kernels."""
    required_kernels = [
        "naif0012.tls",
        "imap_sclk_0000.tsc",
        "imap_wkcp.tf",
        "imap_science_0001.tf",
        "imap_sim_ck_2hr_2secsampling_with_nutation.bc",
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

    # 1 spin/15 seconds; 10 quaternions / spin.
    num_samples = (et_end - et_start) / 15 * 10
    # There were rounding errors when using spiceypy.pxform so np.ceil and np.floor
    # were used to ensure the start and end times were within the ck range.
    et_times = np.linspace(
        np.ceil(et_start * 1e6) / 1e6, np.floor(et_end * 1e6) / 1e6, int(num_samples)
    )

    return et_times


@pytest.mark.parametrize(
    "segment_start_offset, segment_end_offset, quaternion, segment_id",
    [
        ([0], [10], [[1, 0, 0, 0]], [1]),
        ([0, 3600], [10, 7100], [[0, 0, 1, 0], [0, 1, 0, 0]], [1, 2]),
    ],
)
def test_write_pointing_frame_ck(
    segment_start_offset,
    segment_end_offset,
    quaternion,
    segment_id,
    pointing_frame_kernels,
    tmp_path,
):
    """Test coverage for write_pointing_frame_ck"""
    spiceypy.furnsh(pointing_frame_kernels)
    ck_cover = spiceypy.ckcov(
        pointing_frame_kernels[-1],
        SpiceFrame.IMAP_SPACECRAFT,
        True,
        "INTERVAL",
        0,
        "TDB",
    )
    et_start, et_end = spiceypy.wnfetd(ck_cover, 0)
    # Single segment file
    segment_data = np.array(
        [
            (
                spiceypy.sce2c(IMAP_SC_ID, et_start + segment_start_offset[i_seg]),
                spiceypy.sce2c(IMAP_SC_ID, et_start + segment_end_offset[i_seg]),
                quaternion[i_seg],
                segment_id[i_seg],
            )
            for i_seg in range(len(segment_id))
        ],
        dtype=POINTING_SEGMENT_DTYPE,
    )
    pointing_ck = tmp_path / "pointing_ck.bc"
    parent_file = "foo_att.ck"
    write_pointing_frame_ck(pointing_ck, segment_data, parent_file)

    assert pointing_ck.exists()
    spiceypy.furnsh(str(pointing_ck.resolve()))
    # Verify the correct # of segments
    p_cover = spiceypy.ckcov(
        str(pointing_ck), SpiceFrame.IMAP_DPS, True, "INTERVAL", 0, "TDB"
    )
    assert spiceypy.wncard(p_cover) == len(segment_data)

    for i_seg in range(len(segment_id)):
        # Verify that the rotation matrix is as expected
        for et_to_test in np.linspace(
            et_start + segment_start_offset[i_seg],
            et_start + segment_end_offset[i_seg],
            4,
        ):
            rotation_matrix = spiceypy.pxform("ECLIPJ2000", "IMAP_DPS", et_to_test)
            np.testing.assert_allclose(
                rotation_matrix, spiceypy.q2m(segment_data[i_seg]["quaternion"])
            )
    fh = spiceypy.cklpf(str(pointing_ck))
    n_lines, lines, all_lines_returned = spiceypy.dafec(fh, 8, 120)
    assert all_lines_returned
    assert n_lines == 7
    assert parent_file in lines[5]


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
    q_avg = _average_quaternions(et_times)
    rotation_matrix = _create_rotation_matrix(q_avg)
    z_avg = spiceypy.q2m(list(q_avg))[:, 2]

    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    z_avg_expected = np.array([0.4136, 0.9104, 0.0000])

    np.testing.assert_allclose(z_avg, z_avg_expected, atol=1e-4)
    np.testing.assert_allclose(rotation_matrix, rotation_matrix_expected, atol=1e-4)


def get_ck_met_coverage(ck_path: str):
    ck_cover = spiceypy.ckcov(
        ck_path, SpiceFrame.IMAP_SPACECRAFT, True, "INTERVAL", 0, "TDB"
    )
    et_start, et_end = spiceypy.wnfetd(ck_cover, 0)
    met_start = spiceypy.sce2c(IMAP_SC_ID, et_start) * TICK_DURATION
    met_end = spiceypy.sce2c(IMAP_SC_ID, et_end) * TICK_DURATION
    return met_start, met_end


def test_calculate_pointing_attitude_segments(
    spice_test_data_path,
    pointing_frame_kernels,
    tmp_path,
    et_times,
    use_fake_repoint_data_for_time,
):
    """Tests create_pointing_frame function."""
    spiceypy.kclear()
    spiceypy.furnsh(pointing_frame_kernels)

    # Set up the fake repoint data to coincide with the test CK

    # Define 2 repoints:
    #   1. Starts 10 seconds before the input CK start, ends one second
    #      after the CK start
    #   2. Starts one second before the CK ends, ends 10 seconds after the CK ends
    # Result is the pointing starts 1-second after the CK start and ends 1-second
    # before the CK end
    ck_met_start, ck_met_end = get_ck_met_coverage(pointing_frame_kernels[-1])
    use_fake_repoint_data_for_time(
        np.array([ck_met_start - 10, ck_met_end - 1]),
        np.array([ck_met_start + 1, ck_met_end + 10]),
    )

    segment_data = calculate_pointing_attitude_segments(
        spice_test_data_path / "imap_sim_ck_2hr_2secsampling_with_nutation.bc",
    )

    # Nick Dutton's MATLAB code result
    rotation_matrix_expected = np.array(
        [[0.0000, 0.0000, 1.0000], [0.9104, -0.4136, 0.0000], [0.4136, 0.9104, 0.0000]]
    )
    np.testing.assert_almost_equal(
        spiceypy.q2m(segment_data["quaternion"][0]), rotation_matrix_expected, decimal=4
    )

    # Tests error handling when incorrect kernel is loaded.
    spiceypy.furnsh(pointing_frame_kernels)
    with pytest.raises(
        ValueError, match="Error: Expected CK kernel .*badname_kernel.bc"
    ):  # Replace match string with expected error message
        calculate_pointing_attitude_segments(tmp_path / "badname_kernel.bc")


def test_multiple_pointings(
    pointing_frame_kernels, spice_test_data_path, use_fake_repoint_data_for_time
):
    """Tests create_pointing_frame function with multiple pointing kernels."""
    spiceypy.furnsh(pointing_frame_kernels)

    # Define 3 repoints:
    #   1. Starts 10 second before the input CK start, ends one second
    #      after the CK start
    #   2. Starts one hour after CK start, ends 1 second after it starts
    #   3. Starts one second before the CK ends, ends 10 seconds after the CK ends
    # Result is 2 pointings
    ck_met_start, ck_met_end = get_ck_met_coverage(pointing_frame_kernels[-1])
    repoint_start_met = np.array(
        [ck_met_start - 10, ck_met_start + 60 * 60, ck_met_end - 1]
    )
    repoint_end_met = np.array(
        [ck_met_start + 1, ck_met_start + 60 * 60 + 1, ck_met_end + 10]
    )
    use_fake_repoint_data_for_time(repoint_start_met, repoint_end_met)

    segment_data = calculate_pointing_attitude_segments(
        spice_test_data_path / "imap_sim_ck_2hr_2secsampling_with_nutation.bc",
    )

    # Pointings are between repoints, so we expect one less than repoints
    assert len(segment_data["start_sclk_ticks"]) == len(repoint_start_met) - 1

    np.testing.assert_allclose(
        segment_data["start_sclk_ticks"], repoint_end_met[:-1] / TICK_DURATION
    )
    np.testing.assert_allclose(
        segment_data["end_sclk_ticks"], repoint_start_met[1:] / TICK_DURATION
    )
