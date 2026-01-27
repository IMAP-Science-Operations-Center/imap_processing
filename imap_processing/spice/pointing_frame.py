"""Functions for retrieving repointing table data."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spiceypy
from imap_data_access import SPICEFilePath
from numpy.typing import NDArray

from imap_processing.spice import IMAP_SC_ID
from imap_processing.spice.geometry import SpiceBody, SpiceFrame, frame_transform
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


def generate_pointing_attitude_kernel(imap_attitude_cks: list[Path]) -> list[Path]:
    """
    Generate pointing attitude kernel from input IMAP CK kernel.

    Parameters
    ----------
    imap_attitude_cks : list[Path]
        List of the IMAP attitude kernels from which to generate pointing
        attitude.

    Returns
    -------
    pointing_kernel_path : list[Path]
        Location of the new pointing kernels.
    """
    pointing_segments = calculate_pointing_attitude_segments(imap_attitude_cks)
    if len(pointing_segments) == 0:
        raise ValueError("No Pointings covered by input dependencies.")

    # get the start and end yyyy_doy strings
    start_datetime = spiceypy.et2datetime(
        sct_to_et(pointing_segments[0]["start_sclk_ticks"])
    )
    end_datetime = spiceypy.et2datetime(
        sct_to_et(pointing_segments[-1]["end_sclk_ticks"])
    )
    # Use the last ck from sorted list to get the version number. I
    # don't think this will be anything but 1.
    sorted_ck_paths = list(sorted(imap_attitude_cks, key=lambda x: x.name))
    spice_file = SPICEFilePath(sorted_ck_paths[-1].name)
    pointing_kernel_path = (
        sorted_ck_paths[-1].parent / f"imap_dps_"
        f"{start_datetime.strftime('%Y_%j')}_"
        f"{end_datetime.strftime('%Y_%j')}_"
        f"{spice_file.spice_metadata['version']}.ah.bc"
    )
    write_pointing_frame_ck(
        pointing_kernel_path, pointing_segments, [p.name for p in imap_attitude_cks]
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
    pointing_kernel_path: Path, segment_data: np.ndarray, parent_cks: list[str]
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
    parent_cks : list[str]
        Filenames of the CK kernels that the quaternions were derived from.
    """
    comments = [
        "CK FOR IMAP_DPS FRAME",
        "==================================================================",
        "",
        f"Original file name: {pointing_kernel_path.name}",
        f"Creation date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"Parent files: {parent_cks}",
        "",
    ]

    logger.debug(f"Writing pointing attitude kernel: {pointing_kernel_path}")

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
                SpiceFrame.IMAP_DPS.value,
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

    logger.debug(f"Finished writing pointing attitude kernel: {pointing_kernel_path}")


def calculate_pointing_attitude_segments(
    ck_paths: list[Path],
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
    ck_paths : list[pathlib.Path]
        List of CK kernels to use to generate the pointing attitude kernel.

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
    - The latest IMAP frame kernel (imap_###.tf)
    - IMAP DPS frame kernel (imap_science_100.tf)
    - IMAP historical attitude kernel from which the pointing frame kernel will
    be generated.
    """
    logger.info(
        f"Extracting mean spin axes for all Pointings that are"
        f" fully covered by the CK files: {[p.name for p in ck_paths]}"
    )

    # This job relies on the batch starter to provide all the correct CK kernels
    # to cover the time range of the new repoint table.
    # Get the coverage of the CK files storing the earliest start time and
    # latest end time.
    et_start = np.inf
    et_end = -np.inf
    for ck_path in ck_paths:
        ck_cover = spiceypy.ckcov(
            str(ck_path), SpiceBody.IMAP_SPACECRAFT.value, True, "INTERVAL", 0, "TDB"
        )
        num_intervals = spiceypy.wncard(ck_cover)
        individual_ck_start, _ = spiceypy.wnfetd(ck_cover, 0)
        _, individual_ck_end = spiceypy.wnfetd(ck_cover, num_intervals - 1)
        logger.debug(
            f"{ck_path.name} covers time range: ({et_to_utc(individual_ck_start)}, "
            f"{et_to_utc(individual_ck_end)}) in {num_intervals} intervals."
        )
        et_start = min(et_start, individual_ck_start)
        et_end = max(et_end, individual_ck_end)

    logger.info(
        f"CK kernels combined coverage range: "
        f"{(et_to_utc(et_start), et_to_utc(et_end))}, "
    )

    # Get data from the repoint table and convert to Pointings
    repoint_df = get_repoint_data()
    repoint_df["repoint_start_et"] = sct_to_et(
        met_to_sclkticks(repoint_df["repoint_start_met"].values)
    )
    repoint_df["repoint_end_et"] = sct_to_et(
        met_to_sclkticks(repoint_df["repoint_end_met"].values)
    )
    pointing_ids = repoint_df["repoint_id"].values[:-1]
    pointing_start_ets = repoint_df["repoint_end_et"].values[:-1]
    pointing_end_ets = repoint_df["repoint_start_et"].values[1:]

    # Keep only the pointings that are fully covered by the attitude kernels.
    keep_mask = (pointing_start_ets >= et_start) & (pointing_end_ets <= et_end)
    # Filter the pointing data.
    pointing_ids = pointing_ids[keep_mask]
    pointing_start_ets = pointing_start_ets[keep_mask]
    pointing_end_ets = pointing_end_ets[keep_mask]

    n_pointings = len(pointing_ids)
    if n_pointings == 0:
        logger.warning(
            "No Pointings identified based on coverage of CK files. Skipping."
        )

    pointing_segments = np.zeros(n_pointings, dtype=POINTING_SEGMENT_DTYPE)

    for i_pointing in range(n_pointings):
        pointing_segments[i_pointing]["pointing_id"] = pointing_ids[i_pointing]
        pointing_start_et = pointing_start_ets[i_pointing]
        pointing_end_et = pointing_end_ets[i_pointing]
        logger.debug(
            f"Calculating pointing attitude for pointing "
            f"{pointing_segments[i_pointing]['pointing_id']} with time "
            f"range: ({et_to_utc(pointing_start_et)}, {et_to_utc(pointing_end_et)})"
        )

        # Sample at 1Hz
        num_samples = pointing_end_et - pointing_start_et
        # There were rounding errors when using spiceypy.pxform
        # so np.ceil and np.floor were used to ensure the start
        # and end times were within the ck range.
        et_times = np.linspace(
            np.ceil(pointing_start_et * 1e6) / 1e6,
            np.floor(pointing_end_et * 1e6) / 1e6,
            int(num_samples),
        )

        # Get the average spin-axis in HAE coordinates
        z_avg = _mean_spin_axis(et_times)

        # Create a rotation matrix
        rotation_matrix = _create_rotation_matrix(z_avg)

        # Convert the rotation matrix to a quaternion.
        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.m2q
        pointing_segments[i_pointing]["quaternion"] = spiceypy.m2q(rotation_matrix)

        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.sce2c
        # Convert start and end times to SCLK ticks.
        pointing_segments[i_pointing]["start_sclk_ticks"] = spiceypy.sce2c(
            IMAP_SC_ID, pointing_start_et
        )
        pointing_segments[i_pointing]["end_sclk_ticks"] = spiceypy.sce2c(
            IMAP_SC_ID, pointing_end_et
        )

    return pointing_segments


def _mean_spin_axis(et_times: np.ndarray) -> NDArray:
    """
    Compute the mean spin axis for a given time range.

    The mean spin-axis is computed by taking the mean of the spacecraft z-axis
    expressed in HAE Cartesian coordinates at each of the input et_times. The
    mean is computed by finding the mean of each component of the vector across
    time.

    Parameters
    ----------
    et_times : numpy.ndarray
        Array of times between et_start and et_end.

    Returns
    -------
    z_avg : np.ndarray
        Mean spin-axis. Shape is (3,), a single 3D vector (x, y, z).
    """
    # we use a quick and dirty method here for sampling the instantaneous
    # spin-axis.  Depending on how well the kernel input
    # data is built and sampled, there may or may not be aliasing with this
    # approach.  If it turns out that we need to pull the quaternions
    # directly from the CK there are several routines that exist to do this
    # but it's not straight forward.  We'll revisit this if needed.
    z_inertial_hae = frame_transform(
        et_times, np.array([0, 0, 1]), SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.ECLIPJ2000
    )

    # Compute the average spin axis by averaging each component across time
    z_avg = np.mean(z_inertial_hae, axis=0)
    # We don't need to worry about the magnitude being close to zero when
    # normalizing because the instantaneous spin-axes will always be close
    # to the same direction.
    z_avg /= np.linalg.norm(z_avg)

    return z_avg


def _create_rotation_matrix(z_avg: np.ndarray) -> NDArray:
    """
    Create a rotation matrix from the average spin axis.

    Parameters
    ----------
    z_avg : numpy.ndarray
        Average spin-axis that has been normalized to have unit length expressed
        in HAE coordinates.

    Returns
    -------
    rotation_matrix : np.ndarray
        Rotation matrix.
    """
    # y_avg is perpendicular to both z_avg and the HAE Z-axis.
    # Since z_avg will never point anywhere near the HAE Z-axis, this
    # cross-product will always work to define the Pointing Y-axis
    y_avg = np.cross(z_avg, [0, 0, 1])
    y_avg /= np.linalg.norm(y_avg)
    # x_avg is perpendicular to y_avg and z_avg.
    x_avg = np.cross(y_avg, z_avg)
    x_avg /= np.linalg.norm(x_avg)

    # Construct the rotation matrix from x_avg, y_avg, z_avg
    rotation_matrix = np.asarray([x_avg, y_avg, z_avg])

    return rotation_matrix
