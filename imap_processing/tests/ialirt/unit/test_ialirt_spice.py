"""Module to test attitude calculations."""

import numpy as np
import pytest
import spiceypy

from imap_processing.ialirt.l0.ialirt_spice import (
    get_rotation_matrix,
    get_z_axis,
    transform_instrument_vectors_to_inertial,
)
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.kernels import ensure_spice


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


def test_transform_instrument_vectors_to_inertial_no_spice(spice_test_data_path):
    """Tests function transform_instrument_vectors_to_inertial."""

    spiceypy.furnsh(str(spice_test_data_path / "imap_wkcp.tf"))
    sc_inertial_right = np.zeros(3)  # RA = 0
    sc_inertial_decline = np.radians([90, 90, 90])  # Z-axis = [0, 0, 1]

    # Spin phases (0, 90, 180)
    spin_phase = np.radians([0, 90, 180])

    # Unit vector along +X
    instrument_vectors = np.tile(np.array([1.0, 0.0, 0.0]), (3, 1))

    expected = np.array(
        [
            [1.0, 0.0, 0.0],  # No rotation: remains [1, 0, 0]
            [0.0, -1.0, 0.0],  # 90 about +Z: becomes [0, -1, 0]
            [-1.0, 0.0, 0.0],  # 180 about +Z: becomes [-1, 0, 0]
        ]
    )

    et = np.array([0.0, 0.0, 0.0])
    result = transform_instrument_vectors_to_inertial(
        instrument_vectors,
        spin_phase,
        sc_inertial_right,
        sc_inertial_decline,
        et,
        SpiceFrame.IMAP_SPACECRAFT,
        SpiceFrame.IMAP_SPACECRAFT,
    )

    np.testing.assert_allclose(result, expected, atol=1e-8)


@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
# @pytest.mark.external_kernel
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
    et_end = ck_cover[1]
    et = (et_start + et_end) / 2.0

    # Assume IMAP_MAG +X is boresight
    instrument_vector = np.array([[1.0, 0.0, 0.0]])

    # Get RA/Dec of angular momentum vector (Z-axis) from SPICE
    rot_sc_to_j2000 = spiceypy.pxform("IMAP_SPACECRAFT", "ECLIPJ2000", et)
    sc_z_inertial = rot_sc_to_j2000[:, 2]  # SC +Z axis (angular momentum)
    # Convert inertial Z into RA/Dec (radians)
    _, ra, dec = spiceypy.recrad(sc_z_inertial.copy())

    z_axis = get_z_axis(np.array([ra]), np.array([dec]))[0]  # extract the single row

    # Test that our get_z_axis code is returning what SPICE returns.
    np.testing.assert_allclose(
        z_axis,
        sc_z_inertial,
        atol=1e-9,
    )

    # At this timestamp for the attitude kernel.
    spin_phase = np.array([0.0])

    v_manual = transform_instrument_vectors_to_inertial(
        instrument_vector,
        spin_phase,
        np.array([ra]),
        np.array([dec]),
    )

    # SPICE direct transform from instrument frame to inertial
    rot_inst_to_inertial = spiceypy.pxform("IMAP_MAG", "ECLIPJ2000", et)
    v_spice = spiceypy.mxv(rot_inst_to_inertial, instrument_vector[0])
    print("hi")
    np.testing.assert_allclose(
        v_manual[0],
        v_spice,
        atol=1e-9,
    )
