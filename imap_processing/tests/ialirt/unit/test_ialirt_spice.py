"""Module to test attitude calculations."""
import numpy as np

from imap_processing.ialirt.l0.ialirt_spice import get_z_axis, get_x_y_axes, rotate_frame_about_spin_axis


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

    expected = np.array([
        [1.0, 0.0, 0.0],  # RA=0, Dec=0 → +X
        [0.0, 1.0, 0.0], # RA=90°, Dec=0° → +Y
        [0.0, 0.0, 1.0],  # RA=0°, Dec=90° → +Z
    ])

    norms = np.linalg.norm(z_axis, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    assert np.allclose(z_axis, expected, atol=1e-6)


def test_get_x_y_axes():
    """Tests get_x_y_axes function."""

    z_axis = np.array([
        [1.0, 0.0, 0.0],  # RA=0, Dec=0 → +X
        [0.0, 1.0, 0.0],  # RA=90°, Dec=0° → +Y
        [0.0, 0.0, 1.0],  # RA=0°, Dec=90° → +Z
    ])
    x_axis, y_axis = get_x_y_axes(z_axis)

    # Check that the axes are unit vectors.
    assert np.allclose(np.linalg.norm(x_axis, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(y_axis, axis=1), 1.0, atol=1e-6)

    # Check each pair of vectors is 90 degrees apart.
    assert np.allclose(np.sum(x_axis * y_axis, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.sum(x_axis * z_axis, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.sum(y_axis * z_axis, axis=1), 0.0, atol=1e-6)

    # Check cross(X, Y) = Z.
    reconstructed_z = np.cross(x_axis, y_axis)
    assert np.allclose(reconstructed_z, z_axis, atol=1e-6)


def test_get_x_y_axes():
    """Tests get_x_y_axes function."""

    z_axis = np.array([
        [1.0, 0.0, 0.0],  # RA=0, Dec=0 → +X
        [0.0, 1.0, 0.0],  # RA=90°, Dec=0° → +Y
        [0.0, 0.0, 1.0],  # RA=0°, Dec=90° → +Z
    ])
    x_axis, y_axis = get_x_y_axes(z_axis)

    # Check that the axes are unit vectors.
    assert np.allclose(np.linalg.norm(x_axis, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(y_axis, axis=1), 1.0, atol=1e-6)

    # Check each pair of vectors is 90 degrees apart.
    assert np.allclose(np.sum(x_axis * y_axis, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.sum(x_axis * z_axis, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.sum(y_axis * z_axis, axis=1), 0.0, atol=1e-6)

    # Check cross(X, Y) = Z.
    reconstructed_z = np.cross(x_axis, y_axis)
    assert np.allclose(reconstructed_z, z_axis, atol=1e-6)


def test_rotate_frame_about_spin_axis():
    """Tests rotate_frame_about_spin_axis function."""

    z_axis = np.array([
        [1.0, 0.0, 0.0],  # RA=0, Dec=0 → +X
        [0.0, 1.0, 0.0],  # RA=90°, Dec=0° → +Y
        [0.0, 0.0, 1.0],  # RA=0°, Dec=90° → +Z
    ])

    # Rotate 90 degrees (π/2 radians)
    spin_phase = np.pi / 2

    # Get rotation matrix
    R = rotate_frame_about_spin_axis(z_axis, spin_phase)

    # Apply to X-axis
    x = np.array([1, 0, 0])
    x_rot = R @ x

    # Expect X to become Y
    expected = np.array([
        [1.0, 0.0, 0.0],  # Rotating around X leaves X unchanged
        [0.0, 0.0, 1.0],  # Rotating around Y sends X → Z
        [0.0, -1.0, 0.0],  # Rotating around Z sends X → -Y
    ])
    assert np.allclose(x_rot, expected, atol=1e-8)
