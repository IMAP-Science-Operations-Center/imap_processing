"""Module to calculate attitude."""
import numpy as np
from numpy.typing import NDArray
import spiceypy as spice

from imap_processing.spice.geometry import spherical_to_cartesian


def get_z_axis(sc_inertial_right: NDArray, sc_inertial_decline: NDArray) -> NDArray:
    """
    Compute the spacecraft Z-axis (angular momentum direction) in inertial coordinates.

    Parameters
    ----------
    sc_inertial_right : np.ndarray
        Right ascension in radians.

    sc_inertial_decline : np.ndarray
        Declination in radians.

    Returns
    -------
    z_axis : np.ndarray
        Unit vectors of the spacecraft Z-axis (N, 3).
    """

    # Convert right ascension from counts to radians (0-2pi).
    ra_deg = np.degrees(sc_inertial_right)
    # Convert declination from counts to radians (-pi/2 to pi/2).
    dec_deg = np.degrees(sc_inertial_decline)

    # All vectors are unit-length; we only care about direction, not magnitude.
    # So we explicitly set radius r = 1 for all RA/Dec samples.
    r = np.ones_like(ra_deg)

    # Prepare input of shape (N, 3): (r, azimuth=RA, elevation=Dec)
    spherical = np.stack([r, ra_deg, dec_deg], axis=-1)
    z_axis = spherical_to_cartesian(spherical)  # shape: (n, 3)

    return z_axis


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
    v_ref = np.array([0, 1, 0])

    # Detect if z_axis is nearly aligned with v_ref.
    dot_products = np.dot(z_axis, v_ref)
    too_parallel = np.abs(dot_products) > 0.99

    # Use alternate reference vector where needed.
    v_refs = np.tile(v_ref, (z_axis.shape[0], 1))
    v_refs[too_parallel] = np.array([1, 0, 0])

    # Compute a temporary X-axis: perpendicular to both v_ref and z_axis.
    x_temp = np.cross(v_refs, z_axis)
    x_axis = x_temp / np.linalg.norm(x_temp, axis=-1, keepdims=True)

    # Take the cross product to get the Y-axis.
    y_axis = np.cross(z_axis, x_axis)

    return x_axis, y_axis


def rotate_frame_about_spin_axis(
    z_axis: NDArray,
    spin_phase: float
) -> NDArray:
    """
    Rotate a spacecraft frame about the spin axis by the given spin phase angle.

    Parameters
    ----------
    z_axis : NDArray
        Unit vector spacecraft Z-axis.
    spin_phase : float
        Spin phase angle in radians. Positive rotation is right-hand rule about z_axis.

    Returns
    -------
    rot_matrices : NDArray
        Rotation matrix.
    """
    # Rotation matrix to rotate about z_axis by -spin_phase
    rot_matrices = np.stack(
        [spice.axisar(z, -spin_phase) for z in z_axis],
        axis=0
    )

    return rot_matrices

