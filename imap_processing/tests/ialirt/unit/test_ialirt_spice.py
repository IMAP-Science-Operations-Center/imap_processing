"""Module to test attitude calculations."""

import numpy as np
import pytest
import spiceypy

from imap_processing.ialirt.l0.ialirt_spice import (
    compute_total_rotation,
    get_rotation_matrix,
    get_x_y_axes,
    get_z_axis,
    transform_instrument_vectors_to_inertial,
)
from imap_processing.spice.geometry import SpiceFrame, frame_transform


def test_get_z_axis():
    """Tests get_z_axis function."""

    # First case: looking straight along the X-axis.
    # Second case: looking straight along the Y-axis.
    # Third case: looking straight along the Z-axis.
    ra_deg = np.array([0.0, 90.0, 0.0])
    dec_deg = np.array([0.0, 0.0, 90.0])

    z_axis = get_z_axis(ra_deg, dec_deg)

    expected = np.array(
        [
            [1.0, 0.0, 0.0],  # RA=0, Dec=0 → +X
            [0.0, 1.0, 0.0],  # RA=90°, Dec=0° → +Y
            [0.0, 0.0, 1.0],  # RA=0°, Dec=90° → +Z
        ]
    )

    norms = np.linalg.norm(z_axis, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    assert np.allclose(z_axis, expected, atol=1e-6)


def test_get_rotation_matrix():
    """Tests get_rotation_matrix function."""

    z_axis = np.array([[0.0, 0.0, 1.0]])

    # Rotate 90 degrees
    spin_phase = np.array([90])

    # Get rotation matrix
    r = get_rotation_matrix(z_axis, spin_phase)

    # Apply to X-axis
    x = np.array([1, 0, 0])
    x_rot = r @ x

    # Rotating a unit vector pointing along x-axis 90-degrees
    # about z-axis results in a unit vector pointing along y-axis.
    expected = np.array([0, 1.0, 0])
    assert np.allclose(x_rot, expected, atol=1e-8)


def test_get_x_y_axes():
    """Tests get_x_y_axes function."""

    z_axis = np.array([[1.0, 0.0, 0.0]])

    frames = get_x_y_axes(z_axis)
    x_axis = frames[:, 0]
    y_axis = frames[:, 1]
    z_axis = frames[:, 2]

    # Check cross(X, Y) = Z.
    reconstructed_z = np.cross(x_axis, y_axis)
    assert np.allclose(reconstructed_z, z_axis, atol=1e-6)


def test_compute_total_rotation():
    """Test compute_total_rotation function."""

    # Rotation about Z by 90°
    rz_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Rotation about Y by 90°
    ry_90 = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    # Rotation about X by 90°
    rx_90 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    total = compute_total_rotation(
        inertial_frames=np.array([rx_90]),
        spin_rotations=np.array([rz_90]),
        mount_matrix=ry_90,
    )

    # Instrument vector: along +X in instrument frame
    v_instrument = np.array([1.0, 0.0, 0.0])

    # Manually compute expected result:
    intermediate = ry_90 @ v_instrument  # → [0, 0, -1]
    intermediate = rz_90 @ intermediate  # → [0, 0, -1]
    expected = rx_90 @ intermediate  # → [0, 1, 0]

    output_vector = spiceypy.mxv(total[0], v_instrument)

    np.testing.assert_allclose(output_vector, expected, atol=1e-9)


@pytest.mark.external_kernel
def test_transform_instrument_vectors_to_inertial_single(furnish_kernels):
    """Test real-world application of this function."""

    kernels = [
        "imap_science_120.tf",
        "imap_130.tf",
        "naif0012.tls",
        "de440s.bsp",
        "imap_recon_od005_20250925_20251014_v01.bsp",
        "pck00011.tpc",
        "imap_sclk_0036.tsc",
        "imap_2025_283_2025_284_001.ah.bc",
    ]

    with furnish_kernels(kernels):
        # Compare SPICE z-axis with calculated.
        rot_sc_to_j2000 = spiceypy.pxform(
            "IMAP_SPACECRAFT", "ECLIPJ2000", 813433291.0018076
        )
        sc_z_inertial = rot_sc_to_j2000[:, 2]  # SC +Z axis (angular momentum)
        _, ra, dec = spiceypy.recrad(sc_z_inertial.copy())

        z_axis = get_z_axis(np.array([np.degrees(ra)]), np.array([np.degrees(dec)]))[0]
        np.testing.assert_allclose(
            z_axis,
            sc_z_inertial,
            atol=1e-9,
        )

        # Spot check that calculations are similar for vectors.
        instrument_vector = np.array([[-2.525630188, -0.337087161, -4.523789905]])

        v_manual_0 = transform_instrument_vectors_to_inertial(
            instrument_vector,
            np.array([219.5068640401354]),  # spin phase
            np.array([np.degrees(ra)]),  # right ascension
            np.array([np.degrees(dec)]),  # declination
            SpiceFrame.IMAP_MAG_O,
        )

        mago_inertial_vector = frame_transform(
            813433291.0018076,
            instrument_vector,
            from_frame=SpiceFrame.IMAP_MAG_O,
            to_frame=SpiceFrame.ECLIPJ2000,
        )
    np.testing.assert_allclose(
        v_manual_0[0],
        mago_inertial_vector,
        atol=1e-2,
    )
