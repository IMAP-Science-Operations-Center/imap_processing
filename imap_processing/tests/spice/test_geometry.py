"""Tests coverage for imap_processing/spice/geometry.py"""

import numpy as np
import pandas as pd
import pytest
import spiceypy as spice

from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
    basis_vectors,
    cartesian_to_spherical,
    frame_transform,
    get_instrument_spin_phase,
    get_rotation_matrix,
    get_spacecraft_spin_phase,
    get_spacecraft_to_instrument_spin_phase_offset,
    get_spin_data,
    imap_state,
    instrument_pointing,
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


@pytest.fixture()
def fake_spin_data(monkeypatch, spice_test_data_path):
    """Generate fake spin dataframe for testing"""
    fake_spin_path = spice_test_data_path / "fake_spin_data.csv"
    monkeypatch.setenv("SPIN_DATA_FILEPATH", str(fake_spin_path))
    return fake_spin_path


@pytest.mark.parametrize(
    "query_met_times, expected",
    [
        (15, 0.0),  # Scalar test
        (np.array([15.1, 30.1]), np.array([0.1 / 15, 0.1 / 15])),  # Array test
        (np.array([]), None),  # Empty array test
        (np.array([50]), np.array([5 / 15])),  # Single element array test
        # The first spin has thruster firing set, but should return valid value
        (5.0, 5 / 15),
        # Test invalid spin period flag causes nan
        (106.0, np.nan),
        # Test invalid spin phase flag causes nans
        (np.array([121, 122, 123]), np.full(3, np.nan)),
        # Test that invalid spin period causes nans
        (np.array([110, 111]), np.full(2, np.nan)),
        # Test for time in missing spin
        (65, np.nan),
        (np.array([65.1, 66]), np.full(2, np.nan)),
        # Combined test
        (
            np.array([7.5, 30, 61, 75, 106, 121, 136]),
            np.array([0.5, 0, np.nan, 0, np.nan, np.nan, 1 / 15]),
        ),
    ],
)
def test_get_spacecraft_spin_phase(query_met_times, expected, fake_spin_data):
    """Test get_spacecraft_spin_phase() with generated spin data."""
    # Call the function
    spin_phases = get_spacecraft_spin_phase(query_met_times=query_met_times)

    # Test the returned type
    if isinstance(expected, float):
        assert isinstance(spin_phases, float), "Spin phase must be a float."
    elif expected is None:
        assert len(spin_phases) == 0, "Spin phase must be empty."
    else:
        assert spin_phases.shape == expected.shape
    # Test the value
    np.testing.assert_array_almost_equal(spin_phases, expected)


@pytest.mark.parametrize("query_met_times", [-1, 165])
def test_get_spacecraft_spin_phase_value_error(query_met_times, fake_spin_data):
    """Test get_spacecraft_spin_phase() for raising ValueError."""
    with pytest.raises(ValueError, match="Query times"):
        _ = get_spacecraft_spin_phase(query_met_times)


@pytest.mark.usefixtures("use_fake_spin_data_for_time")
def test_get_spin_data(use_fake_spin_data_for_time):
    """Test get_spin_data() with generated spin data."""
    use_fake_spin_data_for_time(453051323.0 - 56120)
    spin_data = get_spin_data()

    (
        np.testing.assert_array_equal(spin_data["spin_number"], np.arange(5761)),
        "One day should have 5,761 records of 15 seconds when including end_met.",
    )
    assert isinstance(spin_data, pd.DataFrame), "Return type must be pandas.DataFrame."

    assert set(spin_data.columns) == {
        "spin_number",
        "spin_start_sec",
        "spin_start_subsec",
        "spin_period_sec",
        "spin_period_valid",
        "spin_phase_valid",
        "spin_period_source",
        "thruster_firing",
        "spin_start_time",
    }, "Spin data must have the specified fields."


@pytest.mark.parametrize(
    "instrument",
    [
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
        SpiceFrame.IMAP_MAG,
    ],
)
def test_get_instrument_spin_phase(instrument, fake_spin_data):
    """Test coverage for get_instrument_spin_phase()"""
    met_times = np.array([7.5, 30, 61, 75, 106, 121, 136])
    expected_nan_mask = np.array([False, False, True, False, True, True, False])
    inst_phase = get_instrument_spin_phase(met_times, instrument)
    assert inst_phase.shape == met_times.shape
    np.testing.assert_array_equal(np.isnan(inst_phase), expected_nan_mask)
    assert np.logical_and(
        0 <= inst_phase[~expected_nan_mask], inst_phase[~expected_nan_mask] < 1
    ).all()


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
        et = np.array([spice.utc2et(et_str) for et_str in et_strings])
        et_arg = et[0] if len(et) == 1 else et
        result = frame_transform(et_arg, position, from_frame, to_frame)
        # check the result shape before modifying for value checking
        assert result.shape == (3,) if len(et) == 1 else (len(et), 3)
        # compare against pure SPICE calculation
        position = np.broadcast_to(position, (len(et), 3))
        result = np.broadcast_to(result, (len(et), 3))
        for spice_et, spice_position, test_result in zip(et, position, result):
            rotation_matrix = spice.pxform(from_frame.name, to_frame.name, spice_et)
            spice_result = spice.mxv(rotation_matrix, spice_position)
            np.testing.assert_allclose(test_result, spice_result, atol=1e-12)


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
            1,
            np.arange(9).reshape((3, 3)),
            SpiceFrame.ECLIPJ2000,
            SpiceFrame.IMAP_HIT,
        )


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
        et = spice.utc2et("2025-09-30T12:00:00.000")
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
        et = spice.utc2et("2025-06-12T12:00:00.000")
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
    et = ensure_spice(spice.utc2et)("2025-09-30T12:00:00.000")
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
        r_spice, colat_spice, slong_spice = spice.recsph(point)

        # Convert SPICE co-latitude to elevation
        el_spice = 90 - np.degrees(colat_spice)
        az_spice = np.degrees(slong_spice)

        # Normalize azimuth to [0, 360]
        az_spice = az_spice % 360

        np.testing.assert_allclose(r, r_spice, atol=1e-5)
        np.testing.assert_allclose(az, az_spice, atol=1e-5)
        np.testing.assert_allclose(el, el_spice, atol=1e-5)


def test_spherical_to_cartesian():
    """Tests spherical_to_cartesian function."""

    azimuth = np.linspace(0, 2 * np.pi, 50)
    elevation = np.linspace(-np.pi / 2, np.pi / 2, 50)
    theta, elev = np.meshgrid(azimuth, elevation)
    r = 1.0

    spherical_points = np.stack(
        (r * np.ones_like(theta).ravel(), theta.ravel(), elev.ravel()), axis=-1
    )

    # Convert elevation to colatitude for SPICE
    colat = np.pi / 2 - spherical_points[:, 2]

    for i in range(len(colat)):
        cartesian_coords = spherical_to_cartesian(np.array([spherical_points[i]]))
        spice_coords = spice.sphrec(r, colat[i], spherical_points[i, 1])

        np.testing.assert_allclose(cartesian_coords[0], spice_coords, atol=1e-5)
