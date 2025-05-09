"""Module to calculate attitude."""

import numpy as np
import spiceypy as spice
from numpy.typing import NDArray

from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform,
    spherical_to_cartesian,
)


def get_z_axis(sc_inertial_right: NDArray, sc_inertial_decline: NDArray) -> NDArray:
    """
    Compute the spacecraft Z-axis (angular momentum direction) in inertial coordinates.

    Parameters
    ----------
    sc_inertial_right : np.ndarray
        Right ascension of the spacecraft spin-axis in radians.

    sc_inertial_decline : np.ndarray
        Declination of the spacecraft spin-axis in radians.

    Returns
    -------
    z_axis : np.ndarray
        Unit vectors of the spacecraft Z-axis (N, 3).
    """
    # Convert right ascension from radians to degrees.
    ra_deg = np.degrees(sc_inertial_right)
    # Convert declination from radians to degrees.
    dec_deg = np.degrees(sc_inertial_decline)

    # All vectors are unit-length; we only care about direction, not magnitude.
    # So we explicitly set radius r = 1 for all RA/Dec samples.
    r = np.ones_like(ra_deg)

    # Prepare input of shape (N, 3): (r, azimuth=RA, elevation=Dec)
    spherical = np.stack([r, ra_deg, dec_deg], axis=-1)
    z_axis = spherical_to_cartesian(spherical)  # shape: (n, 3)

    return z_axis


def get_rotation_matrix(z_axis: NDArray, spin_phase: NDArray) -> NDArray:
    """
    Rotate a spacecraft frame about the spin axis by the given spin phase angle.

    Parameters
    ----------
    z_axis : NDArray
        Unit vector spacecraft Z-axis.
    spin_phase : NDArray
        Spin phase angle in radians.

    Returns
    -------
    rot_matrices : NDArray
        Rotation matrix.

    Notes
    -----
    This matrix acts just like SPICE's pxform(instrument_frame, "IMAP_SPACECRAFT", et).
    A forward rotation that transforms vectors from the instrument's local frame
    to the spacecraft’s rotating frame (URF)
    """
    # Rotation matrix to rotate about z_axis by spin_phase
    rot_matrices = np.array(
        [spice.axisar(z, float(phase)) for z, phase in zip(z_axis, spin_phase)]
    )

    return rot_matrices


def transform_instrument_vectors_to_inertial(
    instrument_vectors: NDArray,
    spin_phase: NDArray,
    sc_inertial_right: NDArray,
    sc_inertial_decline: NDArray,
    et: NDArray,
    instrument_frame: SpiceFrame = SpiceFrame.IMAP_MAG,
    spacecraft_frame: SpiceFrame = SpiceFrame.IMAP_SPACECRAFT,
) -> NDArray:
    """
    Transform instrument-frame vectors into the inertial frame (ECLIPJ2000).
    """
    # Convert RA/Dec → inertial Z-axis
    z_axis = get_z_axis(sc_inertial_right, sc_inertial_decline)

    # Rotation about Z-axis (spin), shape (N, 3, 3)
    rot_spin = get_rotation_matrix(z_axis, spin_phase)

    # Static mount matrix: MAG → SC
    R_mount = spice.pxform("IMAP_MAG", "IMAP_SPACECRAFT", 0.0)

    # SC → inertial
    R_sc_to_inertial = spice.pxform("IMAP_SPACECRAFT", "ECLIPJ2000", et[0])

    # Final transform matrix: (N, 3, 3)
    rot_total = np.array([
        R_sc_to_inertial @ spin @ R_mount for spin in rot_spin
    ])

    # Apply transform
    vectors_inertial = np.array([
        spice.mxv(rot, vec)
        for rot, vec in zip(rot_total, instrument_vectors)
    ])

    return vectors_inertial

