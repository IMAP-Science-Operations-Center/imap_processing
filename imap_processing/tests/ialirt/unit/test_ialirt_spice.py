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
from imap_processing.spice.kernels import ensure_spice


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

    z_axis = np.array(
        [
            [1.0, 0.0, 0.0],  # RA=0, Dec=0 → +X
            [0.0, 1.0, 0.0],  # RA=90°, Dec=0° → +Y
            [0.0, 0.0, 1.0],  # RA=0°, Dec=90° → +Z
        ]
    )

    # Rotate 90 degrees
    spin_phase = np.array([90, 90, 90])

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


def test_get_x_y_axes():
    """Tests get_x_y_axes function."""

    z_axis = np.array(
        [
            [1.0, 0.0, 0.0],  # RA=0, Dec=0 → +X
            [0.0, 1.0, 0.0],  # RA=90°, Dec=0° → +Y
            [0.0, 0.0, 1.0],  # RA=0°, Dec=90° → +Z
        ]
    )
    frames = get_x_y_axes(z_axis)
    x_axis = frames[:, 0, :]
    y_axis = frames[:, 1, :]
    z_axis = frames[:, 2, :]

    # Check that the axes are unit vectors.
    assert np.allclose(np.linalg.norm(x_axis, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(y_axis, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(z_axis, axis=1), 1.0, atol=1e-6)

    # Check each pair of vectors is 90 degrees apart.
    assert np.allclose(np.sum(z_axis * y_axis, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.sum(z_axis * x_axis, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.sum(y_axis * x_axis, axis=1), 0.0, atol=1e-6)

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


@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
@pytest.mark.external_kernel
@ensure_spice
def test_transform_instrument_vectors_to_inertial(
    use_test_metakernel, spice_test_data_path
):
    """Test transform_instrument_vectors_to_inertial function."""

    ck_path = spice_test_data_path / "sim_1yr_imap_attitude.bc"
    id_imap_spacecraft = spiceypy.gipool("FRAME_IMAP_SPACECRAFT", 0, 1)

    ck_cover = spiceypy.ckcov(
        str(ck_path), int(id_imap_spacecraft), True, "INTERVAL", 0, "TDB"
    )

    # Pick midpoint of first coverage interval
    et_start = ck_cover[0]

    # Assume IMAP_MAG +X is boresight
    instrument_vector = np.array([[10.0, 2.0, 3.0]])

    # Get RA/Dec of angular momentum vector (Z-axis) from SPICE
    rot_sc_to_j2000 = spiceypy.pxform("IMAP_SPACECRAFT", "ECLIPJ2000", et_start + 10)
    sc_z_inertial = rot_sc_to_j2000[:, 2]  # SC +Z axis (angular momentum)
    # Convert inertial Z into RA/Dec (radians)
    _, ra, dec = spiceypy.recrad(sc_z_inertial.copy())

    z_axis = get_z_axis(np.array([np.degrees(ra)]), np.array([np.degrees(dec)]))[
        0
    ]  # extract the single row

    # Test that our get_z_axis code is returning what SPICE returns.
    np.testing.assert_allclose(
        z_axis,
        sc_z_inertial,
        atol=1e-9,
    )

    v_manual_0 = transform_instrument_vectors_to_inertial(
        instrument_vector,
        np.array([120.0]),
        np.array([np.degrees(ra)]),
        np.array([np.degrees(dec)]),
    )
    v_manual_1 = transform_instrument_vectors_to_inertial(
        instrument_vector,
        np.array([240.0]),
        np.array([np.degrees(ra)]),
        np.array([np.degrees(dec)]),
    )

    rot_inst_to_inertial_0 = spiceypy.pxform("IMAP_MAG", "ECLIPJ2000", et_start + 10)
    rot_inst_to_inertial_1 = spiceypy.pxform("IMAP_MAG", "ECLIPJ2000", et_start + 20)

    v_spice_0 = spiceypy.mxv(rot_inst_to_inertial_0, instrument_vector[0])
    v_spice_1 = spiceypy.mxv(rot_inst_to_inertial_1, instrument_vector[0])

    np.testing.assert_allclose(
        v_manual_0[0],
        v_spice_0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        v_manual_1[0],
        v_spice_1,
        atol=1e-9,
    )


@pytest.mark.use_test_metakernel("imap_ialirt_sim_metakernel.template")
@pytest.mark.external_kernel
@ensure_spice
def test_no_attitude():
    """Test transform_instrument_vectors_to_inertial function."""

    ra = 0.3653037895099079
    dec = 4.440892098775276e-16

    # Assume IMAP_MAG +X is boresight
    instrument_vector = np.array([[1.0, 0.0, 0.0]])

    # At this timestamp for the attitude kernel.
    spin_phase = np.array([0.0])

    v_manual = transform_instrument_vectors_to_inertial(
        instrument_vector,
        spin_phase,
        np.array([np.degrees(ra)]),
        np.array([np.degrees(dec)]),
    )

    # TODO: Put this into GSE and GSM once we have proper kernels.
    # Example:
    # rotation_ecl_to_gse = spiceypy.pxform("ECLIPJ2000", "GSE", et)
    # v_j2000 = spiceypy.mxv(rotation_ecl_to_gse, v_manual[0])

    assert v_manual is not None
