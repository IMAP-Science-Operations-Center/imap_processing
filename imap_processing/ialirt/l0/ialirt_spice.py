"""Module to simulate SPICE calls with attitude kernels."""

import numpy as np
import spiceypy as spice
from numpy.typing import NDArray

from imap_processing.spice.geometry import (
    SpiceFrame,
    spherical_to_cartesian,
)


def get_z_axis(sc_inertial_right: NDArray, sc_inertial_decline: NDArray) -> NDArray:
    """
    Compute the spacecraft Z-axis in inertial coordinates.

    Parameters
    ----------
    sc_inertial_right : NDArray
        Right ascension of the spacecraft spin-axis in degrees.

    sc_inertial_decline : NDArray
        Declination of the spacecraft spin-axis in degrees.

    Returns
    -------
    z_axis : np.ndarray
        Unit vectors of the spacecraft Z-axis (N, 3).
    """
    # All vectors are unit-length; we only care about direction, not magnitude.
    # So we explicitly set radius r = 1 for all RA/Dec samples.
    r = np.ones_like(sc_inertial_right)

    # Prepare input of shape (N, 3): (r, azimuth=RA, elevation=Dec)
    spherical = np.stack([r, sc_inertial_right, sc_inertial_decline], axis=-1)
    z_axis = spherical_to_cartesian(spherical)  # shape: (n, 3)

    return z_axis


def get_rotation_matrix(axis: NDArray, angle: NDArray) -> NDArray:
    """
    Construct a rotation matrix that rotates vectors by an angle about a specified axis.

    Parameters
    ----------
    axis : NDArray
        Rotation axis.
    angle : NDArray
        Rotation angle, in degrees.

    Returns
    -------
    rot_matrices : NDArray
        Rotation matrices to rotate vectors around Z by spin_phase.
    """
    angle_rad = np.radians(angle)
    rot_matrices = np.array(
        [spice.axisar(z, float(phase)) for z, phase in zip(axis, angle_rad)]
    )

    return rot_matrices


def get_x_y_axes(z_axis: NDArray) -> NDArray:
    """
    Build orthonormal frames from input Z-axis vectors.

    Parameters
    ----------
    z_axis : NDArray
        Array of spacecraft Z-axis unit vectors, shape (N, 3).

    Returns
    -------
    frames : NDArray
        Array of rotation matrices, shape (N, 3, 3).
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

    frames = np.stack([x_axis, y_axis, z_axis], axis=1)

    return frames


def compute_total_rotation(
    inertial_frames: NDArray, spin_rotations: NDArray, mount_matrix: NDArray
) -> NDArray:
    """
    Map instrument vectors to inertial space.

    Parameters
    ----------
    inertial_frames : NDArray
        Spacecraft to inertial rotation matrices (N, 3, 3).
    spin_rotations : NDArray
        Spacecraft spin rotation matrices (N, 3, 3).
    mount_matrix : NDArray
        Matrix for instrument to spacecraft alignment (3, 3).

    Returns
    -------
    total_rotations : NDArray
        Instrument to inertial rotation matrices (N, 3, 3).
    """
    total_rotations = mount_matrix @ spin_rotations @ inertial_frames

    return total_rotations


def transform_instrument_vectors_to_inertial(
    instrument_vectors: NDArray,
    spin_phase: NDArray,
    sc_inertial_right: NDArray,
    sc_inertial_decline: NDArray,
    instrument_frame: SpiceFrame = SpiceFrame.IMAP_MAG,
    spacecraft_frame: SpiceFrame = SpiceFrame.IMAP_SPACECRAFT,
) -> NDArray:
    """
    Rotate instrument vectors into the inertial frame (ECLIPJ2000).

    Parameters
    ----------
    instrument_vectors : NDArray
        Array of instrument-frame vectors, shape (N, 3).
    spin_phase : NDArray
        Spin phase angles (degrees), shape (N,).
    sc_inertial_right : NDArray
        Right ascension of spacecraft spin axis (degrees), shape (N,).
    sc_inertial_decline : NDArray
        Declination of spacecraft spin axis (degrees), shape (N,).
    instrument_frame : SpiceFrame, optional
        SPICE frame of the instrument.
    spacecraft_frame : SpiceFrame, optional
        SPICE frame of the spacecraft.

    Returns
    -------
    vectors : NDArray
        Transformed vectors in the inertial frame (ECLIPJ2000), shape (N, 3).

    Notes
    -----
    Applies: instrument → spacecraft → spun spacecraft → inertial frame.
    """
    # Compute inertial spin axis
    inertial_z_axis = get_z_axis(sc_inertial_right, sc_inertial_decline)

    # Build inertial S/C frames
    inertial_frames = get_x_y_axes(inertial_z_axis)

    # Get spin rotation matrices (around Z) in the spacecraft frame
    # The spin rotation happens in the spacecraft frame, not in inertial frame.
    # In the spacecraft frame, the spin axis is always exactly [0, 0, 1]
    spin_rotations = get_rotation_matrix(
        np.tile([0, 0, 1], (len(spin_phase), 1)), spin_phase
    )

    # Get static mount matrix
    mount_matrix = spice.pxform(instrument_frame.name, spacecraft_frame.name, 0.0)

    # Compute total rotations
    total_rotations = compute_total_rotation(
        inertial_frames, spin_rotations, mount_matrix
    )

    # Apply to instrument vectors
    vectors = np.array(
        [
            spice.mxv(rot.T.copy(), vec)
            for rot, vec in zip(total_rotations, instrument_vectors)
        ]
    )

    return vectors
