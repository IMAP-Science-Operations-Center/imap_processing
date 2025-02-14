from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.spin import (
    get_instrument_spin_phase,
    get_spacecraft_spin_phase,
    get_spin_angle,
    get_spin_data,
    interpolate_spin_data,
)


@pytest.fixture()
def fake_spin_data(monkeypatch, spice_test_data_path):
    """Generate fake spin dataframe for testing"""
    fake_spin_path = spice_test_data_path / "fake_spin_data.csv"
    monkeypatch.setenv("SPIN_DATA_FILEPATH", str(fake_spin_path))
    return fake_spin_path


@pytest.mark.parametrize(
    "query_met_times, expected",
    [
        (
            15,
            [[1, 15, 0, 15.0, 1, 1, 0, 0, 15.0, 0.0]],
        ),  # Scalar test at spin start time
        (
            np.array([15.1, 30.2]),
            [
                [1, 15, 0, 15.0, 1, 1, 0, 0, 15.0, 0.1 / 15],
                [2, 30, 0, 15.0, 1, 1, 0, 0, 30.0, 0.2 / 15],
            ],
        ),  # Array test
    ],
)
def test_interpolate_spin_data(query_met_times, expected, fake_spin_data):
    """Test interpolate_spin_data() with generated spin data."""
    # Call the function
    spin_df = interpolate_spin_data(query_met_times=query_met_times)

    # Test the value
    for i_row, row in enumerate(expected):
        np.testing.assert_array_almost_equal(spin_df.iloc[i_row].tolist(), row)


@pytest.mark.parametrize(
    "query_met_times, expected",
    [
        (15, 0.0),  # Scalar test
        (np.array([15.1, 30.1]), np.array([0.1 / 15, 0.1 / 15])),  # Array test
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
        # Test that this spin phase range [0, 1) is valid which
        # is same as [0, 360) degree angle. At 15 seconds the spacecraft
        # has completed a full spin
        (np.array([0, 15]), np.zeros(2)),
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


@pytest.mark.parametrize(
    "spin_phases, degrees, expected, context",
    [
        (np.arange(0, 1, 0.1), True, np.arange(0, 1, 0.1) * 360, does_not_raise()),
        (
            np.arange(0, 1, 0.1),
            False,
            np.arange(0, 1, 0.1) * 2 * np.pi,
            does_not_raise(),
        ),
        (
            np.array([0, 1]),
            True,
            None,
            pytest.raises(ValueError, match="Spin phases *"),
        ),
        (
            np.array([-1, 0]),
            False,
            None,
            pytest.raises(ValueError, match="Spin phases *"),
        ),
    ],
)
def test_get_spin_angle(spin_phases, degrees, expected, context):
    """Test get_spin_angle() with fake spin phases."""
    with context:
        spin_angles = get_spin_angle(spin_phases, degrees=degrees)
        np.testing.assert_array_equal(spin_angles, expected)


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
