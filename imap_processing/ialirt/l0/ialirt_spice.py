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


def compute_sc_to_inertial_rotation_matrix_from_z(
    z_axis: NDArray,
    spin_phase: NDArray,
) -> NDArray:
    """
    Replace SPICE pxform('IMAP_SPACECRAFT', 'ECLIPJ2000', et)
    using onboard spin axis and spin phase.

    Returns matrix such that: inertial = R @ spacecraft_vector
    """
    R_all = []

    for z, phi in zip(z_axis, spin_phase):
        # Choose reference orthogonal to z
        ref = np.array([0.0, 0.0, 1.0]) if not np.allclose(z, [0, 0, 1.0]) else np.array([1.0, 0.0, 0.0])
        y0 = np.cross(z, ref)
        y0 /= np.linalg.norm(y0)
        x0 = np.cross(y0, z)

        # Spin rotation in XY plane
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        x_rot = cos_phi * x0 + sin_phi * y0
        y_rot = -sin_phi * x0 + cos_phi * y0

        # Columns = spacecraft axes in inertial frame:
        # SPICE: Zsc → X, Ysc → Y, Xsc → Z
        R = np.stack([x_rot, y_rot, z], axis=1)
        R_all.append(R)

    return np.stack(R_all)




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
    #R_sc_to_inertial = spice.pxform("IMAP_SPACECRAFT", "ECLIPJ2000", et[0])

    R_sc_to_inertial_test = compute_sc_to_inertial_rotation_matrix_from_z(
        z_axis,
        spin_phase,
    )
    R_sc_to_inertial = R_sc_to_inertial_test[0]

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

