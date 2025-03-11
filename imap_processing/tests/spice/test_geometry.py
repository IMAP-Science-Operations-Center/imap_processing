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
    get_rotation_matrix,
    get_spacecraft_to_instrument_spin_phase_offset,
    imap_state,
    instrument_pointing,
    solar_longitude,
    spherical_to_cartesian,
)
from imap_processing.spice.kernels import ensure_spice


@pytest.mark.parametrize(
    "et",
    [
        798033670,
        np.linspace(798033670, 798033770),
    ],
)
def test_imap_state(et, use_test_metakernel):
    """Test coverage for imap_state()"""
    state = imap_state(et, observer=SpiceBody.EARTH)
    if hasattr(et, "__len__"):
        np.testing.assert_array_equal(state.shape, (len(et), 6))
    else:
        assert state.shape == (6,)


@pytest.mark.external_kernel()
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_imap_state_ecliptic():
    """Tests retrieving IMAP state in the ECLIPJ2000 frame"""
    state = imap_state(798033670)
    assert state.shape == (6,)


@pytest.mark.parametrize(
    "instrument, expected_offset",
    [
        (SpiceFrame.IMAP_LO, 330 / 360),
        (SpiceFrame.IMAP_HI_45, 255 / 360),
        (SpiceFrame.IMAP_HI_90, 285 / 360),
        (SpiceFrame.IMAP_ULTRA_45, 33 / 360),
        (SpiceFrame.IMAP_ULTRA_90, 210 / 360),
        (SpiceFrame.IMAP_SWAPI, 168 / 360),
        (SpiceFrame.IMAP_IDEX, 90 / 360),
        (SpiceFrame.IMAP_CODICE, 136 / 360),
        (SpiceFrame.IMAP_HIT, 30 / 360),
        (SpiceFrame.IMAP_SWE, 153 / 360),
        (SpiceFrame.IMAP_GLOWS, 127 / 360),
        (SpiceFrame.IMAP_MAG, 0 / 360),
    ],
)
def test_get_spacecraft_to_instrument_spin_phase_offset(instrument, expected_offset):
    """Test coverage for get_spacecraft_to_instrument_spin_phase_offset()"""
    result = get_spacecraft_to_instrument_spin_phase_offset(instrument)
    assert result == expected_offset


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
        "imap_wkcp.tf",
        "imap_science_0001.tf",
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
        for spice_et, spice_position, test_result in zip(et, position, result):
            rotation_matrix = spiceypy.pxform(from_frame.name, to_frame.name, spice_et)
            spice_result = spiceypy.mxv(rotation_matrix, spice_position)
            np.testing.assert_allclose(test_result, spice_result, atol=1e-12)


@pytest.mark.parametrize(
    "spice_frame",
    [
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
        SpiceFrame.ECLIPJ2000,
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


def test_get_rotation_matrix(furnish_kernels):
    """Test coverage for get_rotation_matrix()."""
    kernels = [
        "naif0012.tls",
        "imap_wkcp.tf",
        "imap_science_0001.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels):
        et = spiceypy.utc2et("2025-09-30T12:00:00.000")
        # test input of float
        rotation = get_rotation_matrix(
            et, SpiceFrame.IMAP_IDEX, SpiceFrame.IMAP_SPACECRAFT
        )
        assert rotation.shape == (3, 3)
        # test array of et input
        rotation = get_rotation_matrix(
            np.arange(10) + et, SpiceFrame.IMAP_IDEX, SpiceFrame.IMAP_SPACECRAFT
        )
        assert rotation.shape == (10, 3, 3)


def test_instrument_pointing(furnish_kernels):
    kernels = [
        "naif0012.tls",
        "imap_wkcp.tf",
        "imap_science_0001.tf",
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


@pytest.mark.external_kernel()
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_basis_vectors():
    """Test coverage for basis_vectors()."""
    # This call to SPICE needs to be wrapped with `ensure_spice` so that kernels
    # get furnished automatically
    et = ensure_spice(spiceypy.utc2et)("2025-09-30T12:00:00.000")
    # test input of float
    sc_axes = basis_vectors(et, SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.IMAP_SPACECRAFT)
    np.testing.assert_array_equal(sc_axes, np.eye(3))
    # test array of et input
    et_array = np.arange(10) + et
    sc_axes = basis_vectors(et_array, SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.ECLIPJ2000)
    assert sc_axes.shape == (10, 3, 3)
    # Verify that for each time, the basis vectors are correct
    for et, basis_matrix in zip(et_array, sc_axes):
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
