"""Functions for retrieving repointing table data."""

import logging
import typing
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spiceypy
from imap_data_access import SPICEFilePath
from numpy.typing import NDArray

from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.kernels import ensure_spice
from imap_processing.spice.repoint import get_repoint_data
from imap_processing.spice.time import (
    TICK_DURATION,
    et_to_utc,
    met_to_sclkticks,
    sct_to_et,
)

logger = logging.getLogger(__name__)

POINTING_SEGMENT_DTYPE = np.dtype(
    [
        # sclk ticks are a double precision number of SCLK ticks since the
        # start of the mission (e.g. MET_seconds / TICK_DURATION)
        ("start_sclk_ticks", np.float64),
        ("end_sclk_ticks", np.float64),
        ("quaternion", np.float64, (4,)),
        ("pointing_id", np.uint32),
    ]
)


def generate_pointing_attitude_kernel(imap_attitude_ck: Path) -> list[Path]:
    """
    Generate pointing attitude kernel from input IMAP CK kernel.

    Parameters
    ----------
    imap_attitude_ck : Path
        Location of the IMAP attitude kernel from which to generate pointing
        attitude.

    Returns
    -------
    pointing_kernel_path : list[Path]
        Location of the new pointing kernels.
    """
    pointing_segments = calculate_pointing_attitude_segments(imap_attitude_ck)
    # get the start and end yyyy_doy strings
    # TODO: For now just use the input CK start/end dates. It is possible that
    #    the end date is incorrect b/c the repoint table determines the last
    #    segment in the pointing kernel.
    spice_file = SPICEFilePath(imap_attitude_ck.name)
    pointing_kernel_path = (
        imap_attitude_ck.parent / f"imap_dps_"
        f"{spice_file.spice_metadata['start_date'].strftime('%Y_%j')}_"
        f"{spice_file.spice_metadata['end_date'].strftime('%Y_%j')}_"
        f"{spice_file.spice_metadata['version']}.ah.bc"
    )
    write_pointing_frame_ck(
        pointing_kernel_path, pointing_segments, imap_attitude_ck.name
    )
    return [pointing_kernel_path]


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


def write_pointing_frame_ck(
    pointing_kernel_path: Path, segment_data: np.ndarray, parent_ck: str
) -> None:
    """
    Write a Pointing Frame attitude kernel.

    Parameters
    ----------
    pointing_kernel_path : pathlib.Path
        Location to write the CK kernel.
    segment_data : np.ndarray
        Numpy structured array with the following dtypes:
            ("start_sclk_ticks", np.float64),
            ("end_sclk_ticks", np.float64),
            ("quaternion", np.float64, (4,)),
            ("pointing_id", np.uint32),
    parent_ck : str
        Filename of the CK kernel that the quaternion was derived from.
    """
    id_imap_dps = spiceypy.gipool("FRAME_IMAP_DPS", 0, 1)

    comments = [
        "CK FOR IMAP_DPS FRAME",
        "==================================================================",
        "",
        f"Original file name: {pointing_kernel_path.name}",
        f"Creation date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"Parent file: {parent_ck}",
        "",
    ]

    with open_spice_ck_file(pointing_kernel_path) as handle:
        # Write the comments to the file
        spiceypy.dafac(handle, comments)

        for segment in segment_data:
            # Write the single segment to the file
            # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.ckw02
            spiceypy.ckw02(
                # Handle of an open CK file.
                handle,
                # Start time of the segment.
                segment["start_sclk_ticks"],
                # End time of the segment.
                segment["end_sclk_ticks"],
                # Pointing frame ID.
                int(id_imap_dps),
                # Reference frame.
                SpiceFrame.ECLIPJ2000.name,  # Reference frame
                # Identifier.
                SpiceFrame.IMAP_DPS.name,
                # Number of pointing intervals.
                1,
                # Start times of individual pointing records within segment.
                # Since there is only a single record this is equal to sclk_begtim.
                np.array([segment["start_sclk_ticks"]]),
                # End times of individual pointing records within segment.
                # Since there is only a single record this is equal to sclk_endtim.
                np.array([segment["end_sclk_ticks"]]),  # Single stop time
                # Average quaternion.
                segment["quaternion"],
                # Angular velocity vectors. The IMAP_DPS frame is quasi-inertial
                # for each pointing so each segment has zeros here.
                np.array([0.0, 0.0, 0.0]),
                # The number of seconds per encoded spacecraft clock
                # tick for each interval.
                np.array([TICK_DURATION]),
            )


@typing.no_type_check
@ensure_spice
def calculate_pointing_attitude_segments(
    ck_path: Path,
) -> NDArray:
    """
    Calculate the data for each segment of the DPS_FRAME attitude kernel.

    Each segment corresponds 1:1 with an IMAP pointing. Since the Pointing
    frame is quasi-inertial, the only data needed for each segment are:

    - spacecraft clock start time
    - spacecraft clock end time
    - pointing frame quaternion

    Parameters
    ----------
    ck_path : pathlib.Path
        Location of the CK kernel.

    Returns
    -------
    pointing_segments : numpy.ndarray
        Structured array of data for each pointing. Included fields are:
            ("start_sclk_ticks", np.float64),
            ("end_sclk_ticks", np.float64),
            ("quaternion", np.float64, (4,)),
            ("pointing_id", np.uint32),

    Notes
    -----
    Kernels required to be furnished:

    - Latest NAIF leapseconds kernel (naif0012.tls)
    - The latest IMAP sclk (imap_sclk_NNNN.tsc)
    - The latest IMAP frame kernel (imap_wkcp.tf)
    - IMAP DPS frame kernel (imap_science_0001.tf)
    - IMAP historical attitude kernel from which the pointing frame kernel will
    be generated.
    """
    logger.info(f"Extracting mean spin axes from CK kernel {ck_path.name}")
    # Get IDs.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.gipool
    id_imap_sclk = spiceypy.gipool("CK_-43000_SCLK", 0, 1)

    # Check that the last loaded kernel matches it input kernel name. This ensures
    # that this CK take priority when computing attitude for it's time coverage.
    count = spiceypy.ktotal("ck")
    loaded_ck_kernel, _, _, _ = spiceypy.kdata(count - 1, "ck")
    if str(ck_path) != loaded_ck_kernel:
        raise ValueError(f"Error: Expected CK kernel {ck_path}")

    id_imap_spacecraft = spiceypy.gipool("FRAME_IMAP_SPACECRAFT", 0, 1)

    # Select only the pointings within the attitude coverage.
    ck_cover = spiceypy.ckcov(
        str(ck_path), int(id_imap_spacecraft), True, "INTERVAL", 0, "TDB"
    )
    num_intervals = spiceypy.wncard(ck_cover)
    et_start, _ = spiceypy.wnfetd(ck_cover, 0)
    _, et_end = spiceypy.wnfetd(ck_cover, num_intervals - 1)
    logger.info(
        f"{ck_path.name} contains {num_intervals} intervals with "
        f"start time: {et_to_utc(et_start)}, and end time: {et_to_utc(et_end)}"
    )

    # Get data from the repoint table and filter to only the pointings fully
    # covered by this attitude kernel
    repoint_df = get_repoint_data()
    repoint_df["repoint_start_et"] = sct_to_et(
        met_to_sclkticks(repoint_df["repoint_start_met"].values)
    )
    repoint_df["repoint_end_et"] = sct_to_et(
        met_to_sclkticks(repoint_df["repoint_end_met"].values)
    )
    repoint_df = repoint_df[
        (repoint_df["repoint_end_et"] >= et_start)
        & (repoint_df["repoint_start_et"] <= et_end)
    ]
    n_pointings = len(repoint_df) - 1

    pointing_segments = np.zeros(n_pointings, dtype=POINTING_SEGMENT_DTYPE)

    for i_pointing in range(n_pointings):
        pointing_segments[i_pointing]["pointing_id"] = repoint_df.iloc[i_pointing][
            "repoint_id"
        ]
        pointing_start_et = repoint_df.iloc[i_pointing]["repoint_end_et"]
        pointing_end_et = repoint_df.iloc[i_pointing + 1]["repoint_start_et"]
        logger.debug(
            f"Calculating pointing attitude for pointing "
            f"{pointing_segments[i_pointing]['pointing_id']} with time "
            f"range: ({et_to_utc(pointing_start_et)}, {et_to_utc(pointing_end_et)})"
        )

        # 1 spin/15 seconds; 10 quaternions / spin.
        num_samples = (pointing_end_et - pointing_start_et) / 15 * 10
        # There were rounding errors when using spiceypy.pxform
        # so np.ceil and np.floor were used to ensure the start
        # and end times were within the ck range.
        et_times = np.linspace(
            np.ceil(pointing_start_et * 1e6) / 1e6,
            np.floor(pointing_end_et * 1e6) / 1e6,
            int(num_samples),
        )

        # Get the average quaternions for the pointing
        q_avg = _average_quaternions(et_times)

        # Create a rotation matrix
        rotation_matrix = _create_rotation_matrix(q_avg)

        # Convert the rotation matrix to a quaternion.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.m2q
        pointing_segments[i_pointing]["quaternion"] = spiceypy.m2q(rotation_matrix)

        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.sce2c
        # Convert start and end times to SCLK ticks.
        pointing_segments[i_pointing]["start_sclk_ticks"] = spiceypy.sce2c(
            int(id_imap_sclk), pointing_start_et
        )
        pointing_segments[i_pointing]["end_sclk_ticks"] = spiceypy.sce2c(
            int(id_imap_sclk), pointing_end_et
        )

    return pointing_segments


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


def _create_rotation_matrix(q_avg: np.ndarray) -> NDArray:
    """
    Create a rotation matrix.

    Parameters
    ----------
    q_avg : numpy.ndarray
        Averaged quaternions for the pointing.

    Returns
    -------
    rotation_matrix : np.ndarray
        Rotation matrix.
    """
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
