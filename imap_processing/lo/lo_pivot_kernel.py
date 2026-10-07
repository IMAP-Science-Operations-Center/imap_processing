"""Generate the IMAP-Lo pivot platform attitude kernel (CK)."""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import imap_data_access
import numpy as np
import spiceypy

from imap_processing.cdf.utils import load_cdf
from imap_processing.lo.l1b.lo_l1b import get_median_pivot_angle
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.pointing_frame import (
    POINTING_SEGMENT_DTYPE,
    segment_ck_filename,
    write_constant_attitude_ck,
)
from imap_processing.spice.repoint import get_pointing_times_from_id
from imap_processing.spice.time import (
    et_to_utc,
    met_to_sclkticks,
    met_to_utc,
    sct_to_et,
)

logger = logging.getLogger(__name__)

LO_PIVOT_KERNEL_PREFIX = "imap_lopivot"
LO_PIVOT_KERNEL_EXTENSION = "bc"


def generate_lo_pivot_kernel(
    l1b_nhk_path: Path, repointing: str, minor_version: int
) -> list[Path]:
    """
    Generate the IMAP-Lo pivot platform CK for a single pointing.

    The kernel contains one constant-attitude segment covering the pointing,
    from the end of its repoint maneuver to the start of the next repoint
    maneuver. There is no coverage during repoint maneuvers, which is when
    the pivot platform is moved.

    Parameters
    ----------
    l1b_nhk_path : Path
        Lo L1B NHK file for the pointing.
    repointing : str
        The pointing to generate the kernel for, in the format 'repoint#####'.
    minor_version : int
        Minor version, from the batch command, to use for the output
        kernel filename.

    Returns
    -------
    kernel_paths : list[Path]
        Location of the new Lo pivot kernel.

    Raises
    ------
    ValueError
        If `l1b_nhk_path` is not an NHK file for `repointing`.
    FileExistsError
        If the output kernel already exists.

    Notes
    -----
    Kernels required to be furnished:

    - Latest NAIF leapseconds kernel (naif0012.tls)
    - The latest IMAP sclk (imap_sclk_NNNN.tsc)
    - The latest IMAP frame kernel (imap_###.tf), which defines IMAP_LO_BASE

    The repoint table must also be set (`imap_processing.spice.repoint`), as it
    gives the pointing start and end times.
    """
    repoint_id = imap_data_access.ScienceFilePath(l1b_nhk_path.name).repointing
    if repoint_id is None or f"repoint{repoint_id:05d}" != repointing:
        raise ValueError(f"{l1b_nhk_path.name} is not an NHK file for {repointing}.")

    segment, pivot_angle = calculate_pivot_segment(l1b_nhk_path, repoint_id)

    kernel_filename = segment_ck_filename(
        f"{LO_PIVOT_KERNEL_PREFIX}-{repointing}",
        segment,
        minor_version,
        LO_PIVOT_KERNEL_EXTENSION,
    )
    kernel_path = imap_data_access.SPICEFilePath(kernel_filename).construct_path()
    # open_spice_ck_file would append a duplicate segment to an existing file.
    if kernel_path.exists():
        raise FileExistsError(f"Lo pivot kernel already exists: {kernel_path}")
    kernel_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the kernel in a temporary directory and only publish it once it is
    # complete, so a failed write never leaves a partial kernel at the output
    # path. os.link does not replace an existing file, preserving the
    # no-overwrite behavior even if the kernel appeared during the write.
    with tempfile.TemporaryDirectory(dir=kernel_path.parent) as tmp_dir:
        tmp_kernel_path = Path(tmp_dir) / kernel_path.name
        write_lo_pivot_ck(tmp_kernel_path, segment, pivot_angle, l1b_nhk_path.name)
        os.link(tmp_kernel_path, kernel_path)
    return [kernel_path]


def calculate_pivot_segment(
    l1b_nhk_path: Path, repoint_id: int
) -> tuple[np.ndarray, float]:
    """
    Calculate the data for the single segment of a Lo pivot kernel.

    Parameters
    ----------
    l1b_nhk_path : Path
        Lo L1B NHK file for the pointing.
    repoint_id : int
        Repoint ID of the pointing.

    Returns
    -------
    segment : numpy.ndarray
        Structured array of POINTING_SEGMENT_DTYPE with one element. The
        quaternion rotates vectors from the IMAP_LO_BASE frame into the
        IMAP_LO frame.
    pivot_angle : float
        Pivot angle [degrees] of the pointing.

    Raises
    ------
    ValueError
        If the NHK file has no valid pivot angle samples.
    """
    pointing_start_met, pointing_end_met = get_pointing_times_from_id(repoint_id)
    # Use the same pivot angle as the goodtimes product.
    pivot_angle = get_median_pivot_angle(load_cdf(l1b_nhk_path))
    if np.isnan(pivot_angle):
        raise ValueError(f"No valid pivot angle samples in {l1b_nhk_path.name}.")
    logger.info(
        f"repoint{repoint_id:05d} ({met_to_utc(pointing_start_met)}, "
        f"{met_to_utc(pointing_end_met)}): pivot angle {pivot_angle:.4f} deg"
    )

    segment = np.zeros(1, dtype=POINTING_SEGMENT_DTYPE)
    segment[0]["pointing_id"] = repoint_id
    segment[0]["start_sclk_ticks"] = met_to_sclkticks(pointing_start_met)
    segment[0]["end_sclk_ticks"] = met_to_sclkticks(pointing_end_met)
    segment[0]["quaternion"] = pivot_angle_to_quaternion(pivot_angle)
    return segment, pivot_angle


def pivot_angle_to_quaternion(pivot_angle: float) -> np.ndarray:
    """
    Get the SPICE quaternion rotating IMAP_LO_BASE vectors into IMAP_LO.

    The IMAP_LO frame is the IMAP_LO_BASE frame rotated about its +X axis by
    the pivot angle.

    Parameters
    ----------
    pivot_angle : float
        The pivot angle [degrees].

    Returns
    -------
    quaternion : numpy.ndarray
        SPICE-style quaternion, shape (4,).
    """
    # spiceypy.rotate returns the matrix that rotates the coordinate frame by
    # the angle about the axis, i.e. it maps IMAP_LO_BASE vectors into IMAP_LO.
    rotation_matrix = spiceypy.rotate(np.deg2rad(pivot_angle), 1)
    return np.asarray(spiceypy.m2q(rotation_matrix))


def write_lo_pivot_ck(
    kernel_path: Path,
    segment_data: np.ndarray,
    pivot_angle: float,
    parent_file: str,
) -> None:
    """
    Write the Lo pivot CK, recording the pivot angle in the comments.

    Parameters
    ----------
    kernel_path : pathlib.Path
        Location to write the CK kernel.
    segment_data : numpy.ndarray
        Structured array of POINTING_SEGMENT_DTYPE with one element.
    pivot_angle : float
        Pivot angle [degrees] of the pointing.
    parent_file : str
        Filename of the NHK file the pivot angle was derived from.
    """
    segment = segment_data[0]
    start_utc = et_to_utc(sct_to_et(segment["start_sclk_ticks"]))
    end_utc = et_to_utc(sct_to_et(segment["end_sclk_ticks"]))
    comments = [
        "CK FOR IMAP_LO FRAME (IMAP-LO PIVOT PLATFORM)",
        "==================================================================",
        "",
        f"Original file name: {kernel_path.name}",
        f"Creation date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"Parent files: {[parent_file]}",
        "",
        "The IMAP_LO frame is the IMAP_LO_BASE frame rotated about +X by the",
        "pivot angle, constant over the pointing.",
        "",
        f"Repoint ID: repoint{segment['pointing_id']:05d}",
        f"Pointing start (UTC): {start_utc}",
        f"Pointing end (UTC): {end_utc}",
        f"Pivot angle (deg): {pivot_angle:.4f}",
        "",
    ]

    logger.debug(f"Writing Lo pivot kernel: {kernel_path}")
    write_constant_attitude_ck(
        kernel_path,
        segment_data,
        SpiceFrame.IMAP_LO,
        SpiceFrame.IMAP_LO_BASE,
        comments,
    )
    logger.debug(f"Finished writing Lo pivot kernel: {kernel_path}")
