"""Functions for retrieving repointing table data."""

import logging
import typing
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import spiceypy
from numpy.typing import NDArray

from imap_processing.spice.kernels import ensure_spice
from imap_processing.spice.time import met_to_sclkticks, sct_to_et

logger = logging.getLogger(__name__)


@contextmanager
def open_spice_ck_file(pointing_frame_path: Path) -> Generator[int, None, None]:
    """
    Context manager for handling SPICE CK files.

    Parameters
    ----------
    pointing_frame_path : str
        Path to the CK file.

    Yields
    ------
    handle : int
        Handle to the opened CK file.
    """
    if pointing_frame_path.exists():
        handle = spiceypy.dafopw(str(pointing_frame_path))
    else:
        handle = spiceypy.ckopn(str(pointing_frame_path), "CK", 0)
    try:
        yield handle
    finally:
        spiceypy.ckcls(handle)


@typing.no_type_check
@ensure_spice
def create_pointing_frame(
    pointing_frame_path: Path,
    ck_path: Path,
    repoint_start_met: NDArray,
    repoint_end_met: NDArray,
) -> None:
    """
    Create the pointing frame.

    Parameters
    ----------
    pointing_frame_path : pathlib.Path
        Location of pointing frame kernel.
    ck_path : pathlib.Path
        Location of the CK kernel.
    repoint_start_met : numpy.ndarray
        Start time of the repointing in MET.
    repoint_end_met : numpy.ndarray
        End time of the repointing in MET.

    Notes
    -----
    Kernels required to be furnished:
    "imap_science_0001.tf",
    "imap_sclk_0000.tsc",
    "imap_sim_ck_2hr_2secsampling_with_nutation.bc" or
    "sim_1yr_imap_attitude.bc",
    "imap_wkcp.tf",
    "naif0012.tls"

    Assumptions:
    - The MOC has removed timeframe in which nutation/procession are present.
    TODO: We may come back and have a check for this.
    - The pointing frame kernel is made based on the most recent ck kernel.
    In other words 1:1 ratio.
    """
    # Get IDs.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.gipool
    id_imap_dps = spiceypy.gipool("FRAME_IMAP_DPS", 0, 1)
    id_imap_sclk = spiceypy.gipool("CK_-43000_SCLK", 0, 1)

    # Verify that only ck_path kernel is loaded.
    count = spiceypy.ktotal("ck")
    loaded_ck_kernel, _, _, _ = spiceypy.kdata(count - 1, "ck")

    if count != 1 or str(ck_path) != loaded_ck_kernel:
        raise ValueError(f"Error: Expected CK kernel {ck_path}")

    id_imap_spacecraft = spiceypy.gipool("FRAME_IMAP_SPACECRAFT", 0, 1)

    # Select only the pointings within the attitude coverage.
    ck_cover = spiceypy.ckcov(
        str(ck_path), int(id_imap_spacecraft), True, "INTERVAL", 0, "TDB"
    )
    num_intervals = spiceypy.wncard(ck_cover)
    et_start, _ = spiceypy.wnfetd(ck_cover, 0)
    _, et_end = spiceypy.wnfetd(ck_cover, num_intervals - 1)

    sclk_ticks_start = met_to_sclkticks(repoint_start_met)
    et_start_repoint = sct_to_et(sclk_ticks_start)
    sclk_ticks_end = met_to_sclkticks(repoint_end_met)
    et_end_repoint = sct_to_et(sclk_ticks_end)

    valid_mask = (et_start_repoint >= et_start) & (et_end_repoint <= et_end)
    et_start_repoint = et_start_repoint[valid_mask]
    et_end_repoint = et_end_repoint[valid_mask]

    with open_spice_ck_file(pointing_frame_path) as handle:
        for i in range(len(repoint_start_met)):
            # 1 spin/15 seconds; 10 quaternions / spin.
            num_samples = (et_end_repoint[i] - et_start_repoint[i]) / 15 * 10
            # There were rounding errors when using spiceypy.pxform
            # so np.ceil and np.floor were used to ensure the start
            # and end times were within the ck range.
            et_times = np.linspace(
                np.ceil(et_start_repoint[i] * 1e6) / 1e6,
                np.floor(et_end_repoint[i] * 1e6) / 1e6,
                int(num_samples),
            )

            # Create a rotation matrix
            rotation_matrix = _create_rotation_matrix(et_times)

            # Convert the rotation matrix to a quaternion.
            # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.m2q
            q_avg = spiceypy.m2q(rotation_matrix)

            # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.sce2c
            # Convert start and end times to SCLK.
            sclk_begtim = spiceypy.sce2c(int(id_imap_sclk), et_times[0])
            sclk_endtim = spiceypy.sce2c(int(id_imap_sclk), et_times[-1])

            # Create the pointing frame kernel.
            # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.ckw02
            spiceypy.ckw02(
                # Handle of an open CK file.
                handle,
                # Start time of the segment.
                sclk_begtim,
                # End time of the segment.
                sclk_endtim,
                # Pointing frame ID.
                int(id_imap_dps),
                # Reference frame.
                "ECLIPJ2000",  # Reference frame
                # Identifier.
                "IMAP_DPS",
                # Number of pointing intervals.
                1,
                # Start times of individual pointing records within segment.
                # Since there is only a single record this is equal to sclk_begtim.
                np.array([sclk_begtim]),
                # End times of individual pointing records within segment.
                # Since there is only a single record this is equal to sclk_endtim.
                np.array([sclk_endtim]),  # Single stop time
                # Average quaternion.
                q_avg,
                # 0.0 Angular rotation terms.
                np.array([0.0, 0.0, 0.0]),
                # Rates (seconds per tick) at which the quaternion and
                # angular velocity change.
                np.array([1.0]),
            )


@typing.no_type_check
@ensure_spice
def _average_quaternions(et_times: np.ndarray) -> NDArray:
    """
    Average the quaternions.

    Parameters
    ----------
    et_times : numpy.ndarray
        Array of times between et_start and et_end.

    Returns
    -------
    q_avg : np.ndarray
        Average quaternion.
    """
    aggregate = np.zeros((4, 4))
    for tdb in et_times:
        # we use a quick and dirty method here for grabbing the quaternions
        # from the attitude kernel.  Depending on how well the kernel input
        # data is built and sampled, there may or may not be aliasing with this
        # approach.  If it turns out that we need to pull the quaternions
        # directly from the CK there are several routines that exist to do this
        # but it's not straight forward.  We'll revisit this if needed.

        # Rotation matrix from IMAP spacecraft frame to ECLIPJ2000.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.pxform
        body_rots = spiceypy.pxform("IMAP_SPACECRAFT", "ECLIPJ2000", tdb)
        # Convert rotation matrix to quaternion.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.m2q
        body_quat = spiceypy.m2q(body_rots)

        # Standardize the quaternion so that they may be compared.
        body_quat = body_quat * np.sign(body_quat[0])
        # Aggregate quaternions into a single matrix.
        aggregate += np.outer(body_quat, body_quat)

    # Reference: "On Averaging Rotations".
    # Link: https://link.springer.com/content/pdf/10.1023/A:1011129215388.pdf
    aggregate /= len(et_times)

    # Compute eigen values and vectors of the matrix A
    # Eigenvalues tell you how much "influence" each
    # direction (eigenvector) has.
    # The largest eigenvalue corresponds to the direction
    # that has the most influence.
    # The eigenvector corresponding to the largest
    # eigenvalue points in the direction that has the most
    # combined rotation influence.
    eigvals, eigvecs = np.linalg.eig(aggregate)
    # q0: The scalar part of the quaternion.
    # q1, q2, q3: The vector part of the quaternion.
    q_avg = eigvecs[:, np.argmax(eigvals)]

    return q_avg


def _create_rotation_matrix(et_times: np.ndarray) -> NDArray:
    """
    Create a rotation matrix.

    Parameters
    ----------
    et_times : numpy.ndarray
        Array of times between et_start and et_end.

    Returns
    -------
    rotation_matrix : np.ndarray
        Rotation matrix.
    """
    # Averaged quaternions.
    q_avg = _average_quaternions(et_times)

    # Converts the averaged quaternion (q_avg) into a rotation matrix
    # and get inertial z axis.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.q2m
    z_avg = spiceypy.q2m(list(q_avg))[:, 2]
    # y_avg is perpendicular to both z_avg and the standard Z-axis.
    y_avg = np.cross(z_avg, [0, 0, 1])
    # x_avg is perpendicular to y_avg and z_avg.
    x_avg = np.cross(y_avg, z_avg)

    # Construct the rotation matrix from x_avg, y_avg, z_avg
    rotation_matrix = np.asarray([x_avg, y_avg, z_avg])

    return rotation_matrix
