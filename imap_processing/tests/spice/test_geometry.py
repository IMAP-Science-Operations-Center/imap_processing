"""Tests coverage for imap_processing/spice/geometry.py"""

from unittest import mock

import numpy as np
import pytest
import spiceypy

from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
    basis_vectors,
    cartesian_to_latitudinal,
    cartesian_to_spherical,
    frame_transform,
    frame_transform_az_el,
    get_instrument_mounting_az_el,
    get_rotation_matrix,
    get_spacecraft_to_instrument_spin_phase_offset,
    imap_state,
    instrument_pointing,
    lo_instrument_pointing,
    solar_longitude,
    spherical_to_cartesian,
)


def test_spice_frame_enum(furnish_kernels):
    """Test that the SpiceFrame enum values match imap frames kernel."""
    with furnish_kernels(["imap_130.tf", "imap_science_100.tf"]):
        for frame in SpiceFrame:
            assert frame.value == spiceypy.namfrm(frame.name)


@pytest.mark.parametrize(
    "et",
    [
        798033670,
        np.linspace(798033670, 798033770),
    ],
)
def test_imap_state(et, imap_simple_sim_metakernel):
    """Test coverage for imap_state()"""
    state = imap_state(et, observer=SpiceBody.EARTH)
    if hasattr(et, "__len__"):
        np.testing.assert_array_equal(state.shape, (len(et), 6))
    else:
        assert state.shape == (6,)


@pytest.mark.external_kernel
def test_imap_state_ecliptic(imap_ena_sim_metakernel):
    """Tests retrieving IMAP state in the ECLIPJ2000 frame"""
    state = imap_state(798033670)
    assert state.shape == (6,)


@pytest.mark.parametrize(
    "instrument, expected_az_el",
    [
        # Expected spin-phase offsets based on 7516-0011_drw.pdf
        (SpiceFrame.IMAP_LO_BASE, (60, 0)),  # (330 + 90) % 360 = 60
        # TODO: we need a Lo-pivot CK to test IMAP_LO
        # (SpiceFrame.IMAP_LO, (60, 0)),  # (330 + 90) % 360 = 60
        (SpiceFrame.IMAP_HI_45, (345, -45)),  # 255 + 90 = 345
        (SpiceFrame.IMAP_HI_90, (15, 0)),  # (285 + 90) % 360 = 15
        (SpiceFrame.IMAP_ULTRA_45, (123, -45)),  # 33 + 90 = 123
        (SpiceFrame.IMAP_ULTRA_90, (300, 0)),  # 210 + 90 = 300
        (SpiceFrame.IMAP_SWAPI, (258, 0)),  # 168 + 90 = 258
        (SpiceFrame.IMAP_IDEX, (180, -45)),  # 90 + 90 = 180
        (SpiceFrame.IMAP_CODICE, (226, 0)),  # 136 + 90 = 226
        (SpiceFrame.IMAP_HIT, (120, 0)),  # 30 + 90 = 120
        (SpiceFrame.IMAP_SWE, (243, 0)),  # 153 + 90 = 243
        (SpiceFrame.IMAP_GLOWS, (217, 15)),  # 127 + 90 = 217
        (SpiceFrame.IMAP_MAG_I, (90, 0)),  # 0 + 90 = 90
        (SpiceFrame.IMAP_MAG_O, (90, 0)),  # 0 + 90 = 90
    ],
)
def test_get_instrument_mounting_az_el(
    furnish_kernels, spice_test_data_path, instrument, expected_az_el
):
    """Test coverage for get_instrument_mounting_az_el()"""
    with furnish_kernels([spice_test_data_path / "imap_130.tf"]):
        result = get_instrument_mounting_az_el(instrument)
        # Testing as built angles against nominal. Allow for 0.75 degrees of
        # mounting error.
        np.testing.assert_allclose(result, expected_az_el, atol=0.75)


@pytest.mark.parametrize(
    "instrument",
    [
        # Expected spin-phase offsets based on 7516-0011_drw.pdf
        SpiceFrame.IMAP_LO,
        SpiceFrame.IMAP_HI_45,
        SpiceFrame.IMAP_HI_90,
        SpiceFrame.IMAP_ULTRA_45,
        SpiceFrame.IMAP_ULTRA_90,
        SpiceFrame.IMAP_SWAPI,
        SpiceFrame.IMAP_IDEX,
        SpiceFrame.IMAP_CODICE,
        SpiceFrame.IMAP_HIT,
        SpiceFrame.IMAP_SWE,
        SpiceFrame.IMAP_GLOWS,
        SpiceFrame.IMAP_MAG_I,
        SpiceFrame.IMAP_MAG_O,
    ],
)
def test_get_spacecraft_to_instrument_spin_phase_offset(
    furnish_kernels, spice_test_data_path, instrument
):
    """Test coverage for get_spacecraft_to_instrument_spin_phase_offset()"""
    # Test that the offset is close to SPICE derived mounting azimuth
    with furnish_kernels([spice_test_data_path / "imap_130.tf"]):
        # Lo requires an additional kernel to use the below function. So here,
        # we use the IMAP_LO_BASE frame to verify
        verify_inst = (
            instrument if instrument != SpiceFrame.IMAP_LO else SpiceFrame.IMAP_LO_BASE
        )
        expected = get_instrument_mounting_az_el(verify_inst)[0] / 360
        result = get_spacecraft_to_instrument_spin_phase_offset(instrument)
        np.testing.assert_almost_equal(result, expected, decimal=5)


@pytest.mark.parametrize(
    "et_strings, position, from_frame, to_frame",
    [
        # Single time input, single position input
        (
            ["2025-04-30T12:00:00.000"],
            np.arange(3) + 1,
            SpiceFrame.IMAP_ULTRA_45,
            SpiceFrame.IMAP_DPS,
        ),
        # multiple et and position vectors
        (
            ["2025-04-30T12:00:00.000", "2025-04-30T12:10:00.000"],
            np.arange(6).reshape((2, 3)),
            SpiceFrame.IMAP_HIT,
            SpiceFrame.IMAP_DPS,
        ),
        # multiple et, single position vector
        (
            ["2025-04-30T12:00:00.000", "2025-04-30T12:10:00.000"],
            np.array([0, 0, 1]),
            SpiceFrame.IMAP_SPACECRAFT,
            SpiceFrame.IMAP_DPS,
        ),
        # single et, multiple position vectors
        (
            ["2025-04-30T12:00:00.000"],
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            SpiceFrame.IMAP_SPACECRAFT,
            SpiceFrame.IMAP_DPS,
        ),
    ],
)
def test_frame_transform(et_strings, position, from_frame, to_frame, furnish_kernels):
    """Test transformation of vectors from one frame to another, with the option
    to normalize the result."""
    # This test requires an IMAP attitude kernel and pointing (despun) kernel
    kernels = [
        "naif0012.tls",
        "imap_sclk_0000.tsc",
        "imap_130.tf",
        "imap_science_100.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels):
        # Test single et and position calculation
        et = np.array([spiceypy.utc2et(et_str) for et_str in et_strings])
        et_arg = et[0] if len(et) == 1 else et
        result = frame_transform(et_arg, position, from_frame, to_frame)
        # check the result shape before modifying for value checking.
        # There are 3 cases to consider:

        # 1 event time, multiple position vectors:
        if len(et) == 1 and position.ndim > 1:
            assert result.shape == position.shape
        # multiple event times, single position vector:
        elif len(et) > 1 and position.ndim == 1:
            assert result.shape == (len(et), 3)
        # multiple event times, multiple position vectors (same number of each)
        elif len(et) > 1 and position.ndim > 1:
            assert result.shape == (len(et), 3)

        # compare against pure SPICE calculation.
        # If the result is a single position vector, broadcast it to first.
        if position.ndim == 1:
            position = np.broadcast_to(position, (len(et), 3))
            result = np.broadcast_to(result, (len(et), 3))
        for spice_et, spice_position, test_result in zip(
            et, position, result, strict=False
        ):
            rotation_matrix = spiceypy.pxform(from_frame.name, to_frame.name, spice_et)
            spice_result = spiceypy.mxv(rotation_matrix, spice_position)
            np.testing.assert_allclose(test_result, spice_result, atol=1e-12)


@pytest.mark.parametrize(
    "spice_frame",
    [
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
        SpiceFrame.ECLIPJ2000,
        SpiceFrame.IMAP_GSE,
        SpiceFrame.IMAP_GSM,
        SpiceFrame.IMAP_RTN,
    ],
)
@pytest.mark.parametrize(
    "position",
    [
        np.array([1, 0, 0]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.random.rand(10, 3),
    ],
)
def test_frame_transform_same_frame(position, spice_frame):
    """Test that frame_transform returns position when input/output frames are same."""
    result = frame_transform(0, position, spice_frame, spice_frame)
    assert result is position


def test_frame_transform_exceptions():
    """Test that the proper exceptions get raised when input arguments are invalid."""
    with pytest.raises(
        ValueError, match="Position vectors with one dimension must have 3 elements."
    ):
        frame_transform(
            0, np.arange(4), SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.IMAP_CODICE
        )
    with pytest.raises(ValueError, match="Invalid position shape: "):
        frame_transform(
            np.arange(2),
            np.arange(4).reshape((2, 2)),
            SpiceFrame.ECLIPJ2000,
            SpiceFrame.IMAP_HIT,
        )
    with pytest.raises(
        ValueError,
        match="Mismatch in number of position vectors and Ephemeris times provided.",
    ):
        frame_transform(
            [1, 2],
            np.arange(9).reshape((3, 3)),
            SpiceFrame.ECLIPJ2000,
            SpiceFrame.IMAP_HIT,
        )


@pytest.mark.parametrize(
    "spice_frame",
    [
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
        SpiceFrame.ECLIPJ2000,
    ],
)
def test_frame_transform_az_el_same_frame(spice_frame):
    """Test that frame_transform returns az/el when input/output frames are same."""
    az_el_points = np.array(
        [
            [0, -90],
            [0, 0],
            [0, 89.999999],
            [90, -90],
            [90, 0],
            [90, 89.999999],
            [180, -90],
            [180, 0],
            [180, 89.999999],
            [270, -90],
            [270, 0],
            [270, 89.999999],
            [359.999999, -90],
            [359.999999, 0],
            [359.999999, 89.999999],
            [360, 90],
        ]
    )
    result = frame_transform_az_el(
        0, az_el_points, spice_frame, spice_frame, degrees=True
    )
    np.testing.assert_allclose(result, az_el_points)


def test_frame_transform_az_el_3d_input(furnish_kernels):
    """Test frame_transform_az_el with 3D input array."""
    kernels = [
        "naif0012.tls",
        "imap_001.tf",
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels):
        et = spiceypy.utc2et("2025-06-12T12:00:00.000")

        # Create 3D az_el array with shape (3, 4, 2)
        # This represents 3 energy bins with 4 az/el positions each
        az_el_3d = np.array(
            [
                [[0, 0], [90, 0], [180, 0], [270, 0]],
                [[45, 30], [135, 30], [225, 30], [315, 30]],
                [[0, -45], [90, -45], [180, -45], [270, -45]],
            ]
        )

        result = frame_transform_az_el(
            et, az_el_3d, SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.IMAP_DPS, degrees=True
        )

        # Check that output shape matches input shape
        assert result.shape == az_el_3d.shape

        # Verify by comparing against processing each 2D slice independently
        for i in range(az_el_3d.shape[0]):
            expected_slice = frame_transform_az_el(
                et,
                az_el_3d[i],
                SpiceFrame.IMAP_SPACECRAFT,
                SpiceFrame.IMAP_DPS,
                degrees=True,
            )
            np.testing.assert_allclose(result[i], expected_slice, atol=1e-10)


@pytest.mark.external_kernel
def test_get_rotation_matrix(furnish_kernels):
    """Test coverage for get_rotation_matrix()."""
    kernels = [
        "naif0012.tls",
        "imap_130.tf",
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
        "de440s.bsp",
    ]
    with furnish_kernels(kernels):
        et = spiceypy.utc2et("2025-09-30T12:00:00.000")
        # test input of float
        rotation = get_rotation_matrix(
            et, SpiceFrame.IMAP_IDEX, SpiceFrame.IMAP_SPACECRAFT
        )
        assert rotation.shape == (3, 3)
        assert np.isfinite(rotation).all()
        # test array of et input
        rotation = get_rotation_matrix(
            np.arange(10) + et, SpiceFrame.IMAP_IDEX, SpiceFrame.IMAP_SPACECRAFT
        )
        assert rotation.shape == (10, 3, 3)
        for i in range(10):
            assert np.isfinite(rotation[i]).all()
        rotation = get_rotation_matrix(
            et, SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.IMAP_GSE
        )
        assert rotation.shape == (3, 3)
        assert np.isfinite(rotation).all()


@pytest.mark.external_kernel
def test_get_rotation_matrix_no_transformation_defined_for_et_allowed(furnish_kernels):
    """Test error is swallowed and NaN matrix is returned for undefined SPICE
    transformation when allow_spice_noframeconnect is True in get_rotation_matrix()."""
    kernels = [
        "naif0012.tls",
        "imap_130.tf",
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
        "de440s.bsp",
    ]
    with furnish_kernels(kernels):
        # Midnight is not defined in pointing frame
        et = spiceypy.utc2et("2026-01-01T00:00:00.000")
        rotation = get_rotation_matrix(
            et,
            SpiceFrame.IMAP_MAG_O,
            SpiceFrame.IMAP_DPS,
            allow_spice_noframeconnect=True,
        )
        assert np.isnan(rotation).all()

        # one hour after midnight should have coverage
        ets = np.array([et, et + 3600])
        rotations = get_rotation_matrix(
            ets,
            SpiceFrame.IMAP_MAG_O,
            SpiceFrame.IMAP_DPS,
            allow_spice_noframeconnect=True,
        )
        assert rotations.shape == (2, 3, 3)
        assert np.isnan(rotations[0]).all()
        assert np.isfinite(rotations[1]).all()


@pytest.mark.external_kernel
def test_get_rotation_matrix_no_transformation_defined_for_et_not_allowed(
    furnish_kernels,
):
    """Test error is thrown for undefined SPICE transformation when
    allow_spice_noframeconnect is False (default) in get_rotation_matrix()."""
    kernels = [
        "naif0012.tls",
        "imap_130.tf",
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
        "de440s.bsp",
    ]
    with furnish_kernels(kernels):
        # Midnight is not defined in pointing frame
        et = spiceypy.utc2et("2026-01-01T00:00:00.000")
        with pytest.raises(
            spiceypy.utils.exceptions.SpiceNOFRAMECONNECT,
            match=r"SPICE\(NOFRAMECONNECT\)",
        ):
            get_rotation_matrix(et, SpiceFrame.IMAP_MAG_O, SpiceFrame.IMAP_DPS)


def test_instrument_pointing(furnish_kernels):
    kernels = [
        "naif0012.tls",
        "imap_130.tf",
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels):
        et = spiceypy.utc2et("2025-06-12T12:00:00.000")
        # Single et input
        ins_pointing = instrument_pointing(
            et, SpiceFrame.IMAP_HI_90, SpiceFrame.ECLIPJ2000
        )
        assert ins_pointing.shape == (2,)
        # Multiple et input
        et = np.array([et, et + 100, et + 1000])
        ins_pointing = instrument_pointing(
            et, SpiceFrame.IMAP_HI_90, SpiceFrame.ECLIPJ2000
        )
        assert ins_pointing.shape == (3, 2)
        # Return cartesian coordinates
        ins_pointing = instrument_pointing(
            et, SpiceFrame.IMAP_HI_90, SpiceFrame.ECLIPJ2000, cartesian=True
        )
        assert ins_pointing.shape == (3, 3)


@pytest.mark.parametrize(
    "frame",
    [
        SpiceFrame.IMAP_LO_BASE,
        SpiceFrame.IMAP_HI_45,
        SpiceFrame.IMAP_HI_90,
        SpiceFrame.IMAP_ULTRA_45,
        SpiceFrame.IMAP_ULTRA_90,
        SpiceFrame.IMAP_MAG_I,
        SpiceFrame.IMAP_MAG_O,
        SpiceFrame.IMAP_SWE,
        SpiceFrame.IMAP_SWAPI,
        SpiceFrame.IMAP_CODICE,
        SpiceFrame.IMAP_HIT,
        SpiceFrame.IMAP_IDEX,
        SpiceFrame.IMAP_GLOWS,
    ],
)
def test_instrument_pointing_all_instruments(frame, furnish_kernels):
    """Test the ability to compute instrument pointing for all but Lo."""
    kernels = [
        "naif0012.tls",
        "imap_130.tf",
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels):
        et = spiceypy.utc2et("2025-06-12T12:00:00.000")
        # This only tests functionality, not values
        _ = instrument_pointing(et, frame, SpiceFrame.ECLIPJ2000)


@pytest.mark.parametrize(
    "frame",
    [
        SpiceFrame.IMAP_LO,
        SpiceFrame.IMAP_LO_STAR_SENSOR,
    ],
)
@pytest.mark.xfail(reason="LO and LO_STAR_SENSOR require Lo pivot CK")
def test_instrument_pointing_lo_ck(frame, furnish_kernels):
    """Test calculating Lo pointing."""
    kernels = [
        "naif0012.tls",
        "imap_130.tf",
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels):
        et = spiceypy.utc2et("2025-06-12T12:00:00.000")
        _ = instrument_pointing(et, frame, SpiceFrame.ECLIPJ2000)


@pytest.mark.parametrize(
    "pivot_angle, expected",
    [
        (0, [0.0, 0.0, 1.0]),  # Aligned with SC +Z
        (75, [0.483, 0.837, 0.259]),  # Rotated 75°
        (90, [0.5, 0.866, 0.0]),  # Rotated 90° (perpendicular to SC +Z)
        (105, [0.483, 0.837, -0.259]),  # Rotated 105°
    ],
)
def test_lo_instrument_pointing_pivot_angle(pivot_angle, expected, furnish_kernels):
    kernels = ["imap_130.tf"]
    with furnish_kernels(kernels):
        et = 0  # Use fixed frames, no time-dependent kernels needed

        # Get Lo boresight in spacecraft frame
        boresight_sc = lo_instrument_pointing(
            et, pivot_angle, SpiceFrame.IMAP_SPACECRAFT, cartesian=True
        )

        # Verify angle from spacecraft +Z axis equals pivot angle
        sc_z_axis = np.array([0, 0, 1])
        angle_from_sc_z = np.rad2deg(
            np.arccos(np.clip(np.dot(boresight_sc, sc_z_axis), -1, 1))
        )
        np.testing.assert_allclose(angle_from_sc_z, pivot_angle, atol=1e-8)

        # Verify components match expected values
        np.testing.assert_allclose(boresight_sc, expected, atol=1e-3)

        # Verify boresight is a unit vector
        np.testing.assert_allclose(np.linalg.norm(boresight_sc), 1.0, atol=1e-10)


@pytest.mark.external_kernel
def test_basis_vectors(imap_ena_sim_metakernel):
    """Test coverage for basis_vectors()."""
    et = spiceypy.utc2et("2025-09-30T12:00:00.000")
    # test input of float
    sc_axes = basis_vectors(et, SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.IMAP_SPACECRAFT)
    np.testing.assert_array_equal(sc_axes, np.eye(3))
    # test array of et input
    et_array = np.arange(10) + et
    sc_axes = basis_vectors(et_array, SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.ECLIPJ2000)
    assert sc_axes.shape == (10, 3, 3)
    # Verify that for each time, the basis vectors are correct
    for et, basis_matrix in zip(et_array, sc_axes, strict=False):
        np.testing.assert_array_equal(
            basis_matrix,
            frame_transform(
                et * np.ones(3),
                np.eye(3),
                SpiceFrame.IMAP_SPACECRAFT,
                SpiceFrame.ECLIPJ2000,
            ),
        )


def test_cartesian_to_spherical():
    """Tests cartesian_to_spherical function."""

    step = 0.05
    x = np.arange(-1, 1 + step, step)
    y = np.arange(-1, 1 + step, step)
    z = np.arange(-1, 1 + step, step)
    x, y, z = np.meshgrid(x, y, z)

    cartesian_points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1)

    for point in cartesian_points:
        r, az, el = cartesian_to_spherical(point)
        range, ra, dec = spiceypy.recrad(point)

        np.testing.assert_allclose(r, range, atol=1e-5)
        np.testing.assert_allclose(az, np.degrees(ra), atol=1e-5)
        np.testing.assert_allclose(el, np.degrees(dec), atol=1e-5)


def test_spherical_to_cartesian():
    """Tests spherical_to_cartesian function."""

    azimuth = np.linspace(0, 2 * np.pi, 50)
    elevation = np.linspace(-np.pi / 2, np.pi / 2, 50)
    theta, elev = np.meshgrid(azimuth, elevation)
    r = 1.0

    spherical_points = np.stack(
        (r * np.ones_like(theta).ravel(), theta.ravel(), elev.ravel()), axis=-1
    )
    spherical_points_degrees = np.stack(
        (
            r * np.ones_like(theta).ravel(),
            np.degrees(theta.ravel()),
            np.degrees(elev.ravel()),
        ),
        axis=-1,
    )

    # Convert elevation to colatitude for SPICE
    colat = np.pi / 2 - spherical_points[:, 2]

    cartesian_from_degrees = spherical_to_cartesian(spherical_points_degrees)

    for i in range(len(colat)):
        cartesian_coords = spherical_to_cartesian(
            np.array([spherical_points_degrees[i]])
        )
        spice_coords = spiceypy.sphrec(r, colat[i], spherical_points[i, 1])

        np.testing.assert_allclose(cartesian_coords[0], spice_coords, atol=1e-5)
        np.testing.assert_allclose(cartesian_from_degrees[i], spice_coords, atol=1e-5)


def test_cartesian_to_latitudinal():
    """Test cartesian_to_latitudinal()."""
    # example cartesian coords
    coords = np.ones(3)

    # test with one coord vector
    lat_coords = cartesian_to_latitudinal(coords, degrees=True)
    assert lat_coords.shape == (3,)
    assert lat_coords[1] == 45
    assert lat_coords[2] == 35.264389682754654

    # Test with multiple coord vectors
    coords = np.tile(coords, (10, 1))
    lat_coords = cartesian_to_latitudinal(coords, degrees=True)
    assert lat_coords.shape == (10, 3)


@mock.patch("imap_processing.spice.geometry.imap_state")
def test_solar_longitude(mock_state):
    """Test solar_longitude()."""

    mock_state.side_effect = lambda t, observer: (
        np.ones(6) if (isinstance(t, int)) else np.ones((len(t), 6))
    )
    # example et time
    et = 798033670

    # test for one time interval
    lon = solar_longitude(et, degrees=True)
    assert lon == 45

    # Test with multiple time intervals
    et = np.tile(et, (10, 1))
    lon = solar_longitude(et, degrees=True)
    assert lon.shape == (10,)
