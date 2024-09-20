"""Tests coverage for imap_processing/spice/geometry.py"""

import numpy as np
import pandas as pd
import pytest
import spiceypy as spice

from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
    frame_transform,
    get_spacecraft_spin_phase,
    get_spin_data,
    imap_state,
)


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
@pytest.mark.metakernel("imap_ena_sim_metakernel.template")
def test_imap_state_ecliptic(use_test_metakernel):
    """Tests retrieving IMAP state in the ECLIPJ2000 frame"""
    state = imap_state(798033670)
    assert state.shape == (6,)


@pytest.mark.usefixtures("_set_spin_data_filepath")
@pytest.mark.parametrize(
    "query_met_times, expected_type, expected_length",
    [
        (453051323.0, float, None),  # Scalar test
        (np.array([453051323.0, 453051324.0]), float, 2),  # Array test
        (np.array([]), None, 0),  # Empty array test
        (np.array([453051323.0]), float, 1),  # Single element array test
        # 452995203.0 is a midnight time which should have invalid spin
        # phase and period flags on in the spin data file. The spin phase
        # should be invalid.
        (452995203.0, np.nan, None),
        # Test that five minutes after midnight is also invalid since
        # first 10 minutes after midnight are invalid.
        (np.arange(452995203.0, 452995203.0 + 300), np.nan, 300),
        (
            [453011323.0],
            np.nan,
            1,
        ),  # Test for spin phase that's outside of spin phase range
        (
            453011323.0,
            np.nan,
            None,
        ),  # Test for spin phase that's outside of spin phase range
    ],
)
def test_get_spacecraft_spin_phase(query_met_times, expected_type, expected_length):
    """Test get_spacecraft_spin_phase() with generated spin data."""
    # Call the function
    spin_phases = get_spacecraft_spin_phase(query_met_times=query_met_times)

    # Check the type of the result
    if expected_type is np.nan:
        assert np.isnan(spin_phases).all(), "Spin phase must be NaN."
    elif isinstance(expected_type, float):
        assert isinstance(spin_phases, float), "Spin phase must be a float."

    # If the expected length is None, it means we're testing a scalar
    if expected_length is None:
        assert isinstance(spin_phases, float), "Spin phase must be a float."
    else:
        assert (
            len(spin_phases) == expected_length
        ), f"Spin phase must have length {expected_length} for array input."


@pytest.mark.usefixtures("_set_spin_data_filepath")
def test_get_spin_data():
    """Test get_spin_data() with generated spin data."""

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


def test_frame_transform(furnish_time_kernels, spice_test_data_path):
    """Test transformation of vectors from one frame to another, with the option
    to normalize the result."""
    # This test requires an IMAP attitude kernel and pointing (despun) kernel
    for k in [
        "imap_wkcp.tf",
        "imap_science_0001.tf",
        "sim_1yr_imap_attitude.bc",
        "sim_1yr_imap_pointing_frame.bc",
    ]:
        spice.furnsh(str(spice_test_data_path / k))

    et = spice.utc2et("2025-04-30T12:00:00.000")
    result1 = frame_transform(
        SpiceFrame.IMAP_ULTRA_45, SpiceFrame.IMAP_DPS, et, np.array([1, 0, 0])
    )
    result2 = frame_transform(
        SpiceFrame.IMAP_DPS, SpiceFrame.IMAP_ULTRA_45, et, result1
    )
    np.testing.assert_array_almost_equal(result2, [1, 0, 0])

    vec_result = frame_transform(
        SpiceFrame.IMAP_HI_90,
        SpiceFrame.IMAP_DPS,
        np.array([et, et + 10]),
        np.array([[1, 0, 0], [1, 2, 3]]),
    )

    assert vec_result.shape == (2, 3)


def test_frame_transform_exceptions():
    """Test that the proper exceptions get raised when input arguments are invalid."""
    with pytest.raises(
        ValueError, match="Position vectors with one dimension must have 3 elements."
    ):
        frame_transform(
            SpiceFrame.IMAP_SPACECRAFT, SpiceFrame.IMAP_CODICE, 0, np.arange(4)
        )
    with pytest.raises(
        ValueError,
        match="Ephemeris time must be float when single position vector is provided.",
    ):
        frame_transform(
            SpiceFrame.ECLIPJ2000,
            SpiceFrame.IMAP_HIT,
            np.asarray(0),
            np.array([1, 0, 0]),
        )
    with pytest.raises(ValueError, match="Invalid position shape: "):
        frame_transform(
            SpiceFrame.ECLIPJ2000,
            SpiceFrame.IMAP_HIT,
            np.arange(2),
            np.arange(4).reshape((2, 2)),
        )
    with pytest.raises(
        ValueError,
        match="Mismatch in number of position vectors and Ephemeris times provided.",
    ):
        frame_transform(
            SpiceFrame.ECLIPJ2000,
            SpiceFrame.IMAP_HIT,
            np.arange(2),
            np.arange(9).reshape((3, 3)),
        )
