"""Module to calculate attitude."""

import numpy as np
import spiceypy as spice
from numpy.typing import NDArray

from imap_processing.spice.geometry import (
    SpiceFrame,
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
    z_axis = get_z_axis(sc_inertial_right, sc_inertial_decline)

    # Construct orthonormal S/C frame in inertial space
    inertial_frames = []
    for z in z_axis:
        # Pick a reference not parallel to z
        ref = (
            np.array([0.0, 0.0, 1.0])
            if not np.allclose(z, [0, 0, 1.0])
            else np.array([1.0, 0.0, 0.0])
        )
        y = np.cross(z, ref)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        R_sc_to_inertial = np.stack([x, y, z], axis=1)
        inertial_frames.append(R_sc_to_inertial)
    inertial_frames = np.array(inertial_frames)

    # Rotation about z by spin phase (in spacecraft XY plane)
    rot_spin = get_rotation_matrix(np.tile([0, 0, 1], (len(spin_phase), 1)), spin_phase)

    # Static mount matrix from instrument to spacecraft
    R_mount = spice.pxform(instrument_frame.name, spacecraft_frame.name, 0.0)

    # Final transform: inertial = R_sc @ spin @ R_mount @ instrument_vector
    rot_total = np.array(
        [R_sc @ spin @ R_mount for R_sc, spin in zip(inertial_frames, rot_spin)]
    )

    # Apply to instrument vectors
    vectors_inertial = np.array(
        [spice.mxv(rot, vec) for rot, vec in zip(rot_total, instrument_vectors)]
    )

    return vectors_inertial
