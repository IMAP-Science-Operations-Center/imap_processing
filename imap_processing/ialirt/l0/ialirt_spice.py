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


def transform_instrument_vectors_to_urf(
    instrument_vectors: NDArray,
    spin_phase: NDArray,
    sc_inertial_right: NDArray,
    sc_inertial_decline: NDArray,
    et: np.ndarray,
) -> NDArray:
    """
    Transform instrument-frame vectors into the spacecraft URF frame.

    Parameters
    ----------
    instrument_vectors : np.ndarray
        Vectors in the instrument frame. Shape: (N, 3).
    spin_phase : np.ndarray
        Spin phase angle(s) in radians. Shape: (N,).
    sc_inertial_right : np.ndarray
        Spacecraft right ascension in radians. Shape: (N,).
    sc_inertial_decline : np.ndarray
        Spacecraft declination in radians. Shape: (N,).
    et : np.ndarray
        Ephemeris time.

    Returns
    -------
    vectors_urf : np.ndarray
        Vectors in the spacecraft URF frame. Shape: (N, 3).

    Notes
    -----
    URF = Unrotated Reference Frame.
    It is a spacecraft-fixed frame that rotates with the spacecraft.
    """
    z_axis = get_z_axis(sc_inertial_right, sc_inertial_decline)
    rot_matrices = get_rotation_matrix(z_axis, spin_phase)
    vectors_urf = get_instrument_vector(
        et,
        instrument_vectors,
    )

    vectors_inertial = np.array(
        [spice.mxv(r.T, v) for r, v in zip(rot_matrices, vectors_urf)]
    )

    return vectors_inertial


def get_instrument_vector(
    et: np.ndarray,
    vector: np.ndarray,
    instrument_frame: SpiceFrame = SpiceFrame.IMAP_MAG,
    spacecraft_frame: SpiceFrame = SpiceFrame.IMAP_SPACECRAFT,
) -> np.ndarray:
    """
    Get the vectors wrt the spacecraft.

    Parameters
    ----------
    et : np.ndarray
        Ephemeris time.
    vector : np.ndarray
        Vector in the instrument frame.
    instrument_frame : SpiceFrame
        Instrument frame.
    spacecraft_frame : SpiceFrame
        Spacecraft frame.

    Returns
    -------
    vector_urf : np.ndarray
        Transformed vector(s) in the spacecraft frame.
    """
    # Instrument frame → SC frame (URF)
    vector_urf = frame_transform(
        et=et,
        position=vector,
        from_frame=instrument_frame,
        to_frame=spacecraft_frame,
    )

    return vector_urf
