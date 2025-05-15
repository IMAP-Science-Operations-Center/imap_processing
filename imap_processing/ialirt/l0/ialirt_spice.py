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


def build_sc_frame_in_inertial(z_axis: NDArray) -> NDArray:
    """
    Create spacecraft orthonormal frame in inertial space for each z-axis.
    """
    frames = []
    for z in z_axis:
        ref = np.array([0.0, 0.0, 1.0]) if not np.allclose(z, [0, 0, 1.0]) else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, ref)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        frames.append(np.stack([x, y, z], axis=1))
    return np.array(frames)


def get_x_y_axes(z_axis: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute X and Y vectors that are perpendicular to Z and to each other.
    Parameters
    ----------
    z_axis : NDArray
        Array of shape (N, 3).
    Returns
    -------
    x_axis : NDArray
        Array of shape (N, 3) perpendicular to z_axis.
    y_axis : NDArray
        Array of shape (N, 3) perpendicular to z_axis.
    """
    # Pick a fixed reference vector.
    v_ref = np.array([0, 0, 1])

    # Detect if z_axis is nearly aligned with v_ref.
    dot_products = np.dot(z_axis, v_ref)
    too_parallel = np.abs(dot_products) > 0.99

    # Use alternate reference vector where needed.
    v_refs = np.tile(v_ref, (z_axis.shape[0], 1))
    v_refs[too_parallel] = np.array([1, 0, 0])

    # Compute a temporary Y-axis: perpendicular to both z_axis and v_ref.
    y_temp = np.cross(z_axis, v_refs)
    # Make it a unit vector.
    y_axis = y_temp / np.linalg.norm(y_temp, axis=-1, keepdims=True)

    # Take the cross product to get the X-axis.
    x_axis = np.cross(y_axis, z_axis)

    frames = np.stack([y_axis, z_axis, x_axis], axis=1)

    return frames


def get_instrument_mount_matrix(instrument_frame: SpiceFrame, spacecraft_frame: SpiceFrame) -> NDArray:
    """
    Get static instrument-to-spacecraft rotation matrix.
    """
    return spice.pxform(instrument_frame.name, spacecraft_frame.name, 0.0)


def compute_total_rotation(
    inertial_frames: NDArray,
    spin_rotations: NDArray,
    mount_matrix: NDArray
) -> NDArray:
    """
    Compute full rotation matrices from instrument to inertial frame.
    """
    return np.array([R_sc @ spin @ mount_matrix for R_sc, spin in zip(inertial_frames, spin_rotations)])


def apply_rotations_to_vectors(rotations: NDArray, vectors: NDArray) -> NDArray:
    """
    Apply rotation matrices to instrument vectors.
    """
    return np.array([spice.mxv(rot, vec) for rot, vec in zip(rotations, vectors)])


def transform_instrument_vectors_to_inertial(
    instrument_vectors: NDArray,
    spin_phase: NDArray,
    sc_inertial_right: NDArray,
    sc_inertial_decline: NDArray,
    instrument_frame: SpiceFrame = SpiceFrame.IMAP_MAG,
    spacecraft_frame: SpiceFrame = SpiceFrame.IMAP_SPACECRAFT,
) -> NDArray:
    """
    Transform instrument-frame vectors into the inertial frame (ECLIPJ2000).
    """
    # Step 1: compute spin axis
    z_axis = get_z_axis(sc_inertial_right, sc_inertial_decline)

    # Step 2: build inertial S/C frames
    inertial_frames = build_sc_frame_in_inertial(z_axis)

    inertial_frames2 = get_x_y_axes(z_axis)

    # Step 3: get spin rotation matrices (around Z)
    spin_rotations = get_rotation_matrix(np.tile([0, 0, 1], (len(spin_phase), 1)), spin_phase)

    # Step 4: get static mount matrix
    mount_matrix = get_instrument_mount_matrix(instrument_frame, spacecraft_frame)

    # Step 5: compute total rotations
    total_rotations = compute_total_rotation(inertial_frames, spin_rotations, mount_matrix)

    # Step 6: apply to instrument vectors
    return apply_rotations_to_vectors(total_rotations, instrument_vectors)

