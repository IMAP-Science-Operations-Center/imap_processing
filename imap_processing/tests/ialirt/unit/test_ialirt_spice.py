"""Module to test attitude calculations."""

import numpy as np

from imap_processing.ialirt.l0.ialirt_spice import (
    get_rotation_matrix,
    get_z_axis,
    transform_instrument_vectors_to_urf,
)


def test_get_z_axis():
    """Tests get_z_axis function."""

    # First case: looking straight along the X-axis.
    # Second case: looking straight along the Y-axis.
    # Third case: looking straight along the Z-axis.
    ra_deg = np.array([0.0, 90.0, 0.0])
    dec_deg = np.array([0.0, 0.0, 90.0])

    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    z_axis = get_z_axis(ra_rad, dec_rad)

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

    z_axis = np.array(
        [
            [1.0, 0.0, 0.0],  # RA=0, Dec=0 → +X
            [0.0, 1.0, 0.0],  # RA=90°, Dec=0° → +Y
            [0.0, 0.0, 1.0],  # RA=0°, Dec=90° → +Z
        ]
    )

    # Rotate 90 degrees (π/2 radians)
    spin_phase = np.array([np.pi / 2, np.pi / 2, np.pi / 2])

    # Get rotation matrix
    r = get_rotation_matrix(z_axis, spin_phase)

    # Apply to X-axis
    x = np.array([1, 0, 0])
    x_rot = r @ x

    expected = np.array(
        [
            [1.0, 0.0, 0.0],  # Rotating around X leaves X unchanged
            [0.0, 0.0, -1.0],  # Rotating around Y sends X → -Z
            [0.0, 1.0, 0.0],  # Rotating around Z sends X → Y
        ]
    )
    assert np.allclose(x_rot, expected, atol=1e-8)


def test_transform_instrument_vectors_to_urf():
    """Tests function transform_instrument_vectors_to_urf."""

    sc_inertial_right = np.zeros(3)  # RA = 0
    sc_inertial_decline = np.radians([90, 90, 90])  # Z-axis = [0, 0, 1]

    # Spin phases (0, 90, 180)
    spin_phase = np.radians([0, 90, 180])

    # Unit vector along +X
    instrument_vectors = np.tile(np.array([1.0, 0.0, 0.0]), (3, 1))

    expected = np.array(
        [
            [1.0, 0.0, 0.0],  # No rotation: remains [1, 0, 0]
            [0.0, 1.0, 0.0],  # 90 about +Z: becomes [0, 1, 0]
            [-1.0, 0.0, 0.0],  # 180 about +Z: becomes [-1, 0, 0]
        ]
    )

    result = transform_instrument_vectors_to_urf(
        instrument_vectors, spin_phase, sc_inertial_right, sc_inertial_decline
    )

    np.testing.assert_allclose(result, expected, atol=1e-8)
