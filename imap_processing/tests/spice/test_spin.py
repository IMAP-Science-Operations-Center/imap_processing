from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imap_processing.spice import config, spin
from imap_processing.spice.geometry import SpiceFrame


@pytest.fixture
def fake_spin_data(spice_test_data_path, use_test_spin_data_csv):
    """Generate fake spin dataframe for testing"""
    fake_spin_path = spice_test_data_path / "fake_spin_data.csv"
    use_test_spin_data_csv([fake_spin_path])
    return fake_spin_path


def test_set_spin_table_paths(monkeypatch):
    """Test coverage for set_spin_table_paths function."""
    # Use monkeypatch here to make sure any side effects of calling the setter
    # get undone after this test
    monkeypatch.setattr(config, "_spin_table_paths", [])
    assert config._spin_table_paths == []
    spin_paths = [
        Path("/path/to/fake_spin_data0.csv"),
        Path("/path/to/fake_spin_data1.csv"),
    ]
    spin.set_global_spin_table_paths(spin_paths)
    np.testing.assert_array_equal(config._spin_table_paths, spin_paths)


@pytest.mark.parametrize(
    "query_met_times, expected",
    [
        (
            15,
            [
                [
                    1,
                    15,
                    0,
                    "2024-04-11 00:00:15.000000",
                    15.0,
                    True,
                    True,
                    0,
                    False,
                    15.0,
                    15.0,  # actual_spin_period
                    0.0,
                ]
            ],
        ),  # Scalar test at spin start time
        (
            np.array([15.1, 30.2]),
            [
                [
                    1,
                    15,
                    0,
                    "2024-04-11 00:00:15.000000",
                    15.0,
                    True,
                    True,
                    0,
                    False,
                    15.0,
                    15.0,  # actual_spin_period
                    0.1 / 15,
                ],
                [
                    2,
                    30,
                    0,
                    "2024-04-11 00:00:30.000000",
                    15.0,
                    True,
                    True,
                    0,
                    False,
                    30.0,
                    15.0,  # actual_spin_period
                    0.2 / 15,
                ],
            ],
        ),  # Array test
    ],
)
def test_interpolate_spin_data(query_met_times, expected, fake_spin_data):
    """Test interpolate_spin_data() with generated spin data."""
    # Call the function
    spin_df = spin.interpolate_spin_data(query_met_times=query_met_times)

    # Test the value
    for i_row, row in enumerate(expected):
        pd.testing.assert_series_equal(
            spin_df.iloc[i_row], pd.Series(row), check_index=False, check_names=False
        )


@pytest.mark.parametrize(
    "met_time,spin_number_expected", [(0, 0), (14.9, 0), (15.1, 1), (32, 2)]
)
def test_get_spin_number(fake_spin_data, met_time, spin_number_expected):
    """Test get_spin_number() with generated spin data."""
    # Call the function
    spin_number = spin.get_spin_number(met_time=met_time)

    # Test the value
    assert spin_number == spin_number_expected


@pytest.mark.parametrize(
    "query_met_times, expected",
    [
        (15, 0.0),  # Scalar test
        (np.array([15.1, 30.1]), np.array([0.1 / 15, 0.1 / 15])),  # Array test
        # Query time 50 is in spin 3 (45-60), uses spin_period_sec=15 (spin 4 missing)
        (np.array([50]), np.array([5 / 15])),  # Single element array test
        # The first spin has thruster firing set, but should return valid value
        (5.0, 5 / 15),
        # Test invalid spin period flag causes nan
        (106.0, np.nan),
        # Test invalid spin phase flag causes nans
        (np.array([121, 122, 123]), np.full(3, np.nan)),
        # Test that invalid spin period causes nans
        (np.array([110, 111]), np.full(2, np.nan)),
        # Test for time in gap (spin 4 missing) - invalid because (65-45)/15 > 1
        (65, np.nan),
        (np.array([65.1, 66]), np.full(2, np.nan)),
        # Combined test
        (
            np.array([7.5, 30, 61, 75, 106, 121, 136]),
            # 61 is in spin 3, uses spin_period_sec=15, (61-45)/15 > 1 → invalid
            np.array([0.5, 0, np.nan, 0, np.nan, np.nan, 1 / 15]),
        ),
        # Test that this spin phase range [0, 1) is valid which
        # is same as [0, 360) degree angle. At 15 seconds the spacecraft
        # has completed a full spin
        (np.array([0, 15]), np.zeros(2)),
        # Test gap between estimated spin end and actual next spin start
        # Spin 9: start=135, spin_period_sec=14.5, estimated_end=149.5
        # Spin 10: start=150 (actual start is 0.5s after estimated end)
        # Query at 149.7 should be valid: (149.7-135)/15 = 0.98 < 1
        (149.7, 14.7 / 15),
    ],
)
def test_get_spacecraft_spin_phase(query_met_times, expected, fake_spin_data):
    """Test get_spacecraft_spin_phase() with generated spin data."""
    # Call the function
    spin_phases = spin.get_spacecraft_spin_phase(query_met_times=query_met_times)

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
        spin_angles = spin.get_spin_angle(spin_phases, degrees=degrees)
        np.testing.assert_array_equal(spin_angles, expected)


@pytest.mark.parametrize("query_met_times", [-1, 181])
def test_get_spacecraft_spin_phase_value_error(query_met_times, fake_spin_data):
    """Test get_spacecraft_spin_phase() for raising ValueError."""
    with pytest.raises(ValueError, match="Query times"):
        _ = spin.get_spacecraft_spin_phase(query_met_times)


@pytest.mark.usefixtures("use_fake_spin_data_for_time")
def test_get_spin_data(use_fake_spin_data_for_time):
    """Test get_spin_data() with generated spin data."""
    use_fake_spin_data_for_time(453051323.0 - 56120)
    spin_data = spin.get_spin_data()

    (
        np.testing.assert_array_equal(spin_data.index, np.arange(5761)),
        "One day should have 5,761 records of 15 seconds when including end_met.",
    )
    assert isinstance(spin_data, pd.DataFrame), "Return type must be pandas.DataFrame."

    assert set(spin_data.columns) == {
        "spin_number",
        "spin_start_sec_sclk",
        "spin_start_subsec_sclk",
        "spin_start_utc",
        "spin_period_sec",
        "spin_period_valid",
        "spin_phase_valid",
        "spin_period_source",
        "thruster_firing",
        "spin_start_met",
        "actual_spin_period",
    }, "Spin data must have the specified fields."


def test_get_spin_table_merge(tmp_path, use_test_spin_data_csv):
    """Test that get_spin_table() merges spin tables correctly."""
    columns = [
        "spin_number",
        "spin_start_sec_sclk",
        "spin_start_subsec_sclk",
        "spin_start_utc",
        "spin_period_sec",
        "spin_period_valid",
        "spin_phase_valid",
        "spin_phase_source",
        "thruster_firing",
    ]
    # Table 1 is missing spin # 2
    table1_data = [
        [0, 0, 0, "2025-05-01 00:00:00.000", 15, 1, 1, 0, 0],
        [1, 15, 0, "2025-05-01 00:00:15.000", 15, 1, 1, 0, 0],
        [3, 45, 0, "2025-05-01 00:00:45.000", 15, 1, 1, 0, 0],
        [4, 60, 0, "2025-05-01 00:01:00.000", 15, 1, 1, 0, 0],
    ]
    table1_path = tmp_path / "imap_2025_100_2025_101_01.spin.csv"
    pd.DataFrame.from_records(table1_data, columns=columns, index=columns[0]).to_csv(
        table1_path
    )
    # Table 2 fills in spin #2 and changes values for spin #3
    table2_data = [
        [2, 30, 0, "2025-05-01 00:00:30.000", 15.1, 1, 1, 0, 0],
        [3, 45, 1e5, "2025-05-01 00:00:45.100", 14.9, 1, 1, 0, 0],
        [5, 75, 0, "2025-05-01 00:01:15.000", 15, 1, 1, 0, 0],
        [6, 90, 0, "2025-05-01 00:01:30.000", 15, 1, 1, 0, 0],
    ]
    table2_path = tmp_path / "imap_2025_101_2025_102_01.spin.csv"
    pd.DataFrame.from_records(table2_data, columns=columns, index=columns[0]).to_csv(
        table2_path
    )
    # Intentionally set table 2 as the first
    use_test_spin_data_csv([table2_path, table1_path])
    combined_df = spin.get_spin_data()
    assert len(combined_df) == 7
    # Check that table 2 fills missing spin #2
    assert combined_df.iloc[2]["spin_start_sec_sclk"] == 30
    # Check that table 2 overrides spin #3 values
    assert combined_df.loc[3]["spin_start_subsec_sclk"] == table2_data[1][2]


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
        SpiceFrame.IMAP_MAG_I,
        SpiceFrame.IMAP_MAG_O,
    ],
)
def test_get_instrument_spin_phase(
    instrument, fake_spin_data, furnish_kernels, spice_test_data_path
):
    """Test coverage for get_instrument_spin_phase()"""
    met_times = np.array([7.5, 30, 61, 75, 106, 121, 136])
    # Time 61 is in missing spin gap, should be invalid
    expected_nan_mask = np.array([False, False, True, False, True, True, False])
    with furnish_kernels([spice_test_data_path / "imap_130.tf"]):
        inst_phase = spin.get_instrument_spin_phase(met_times, instrument)
    assert inst_phase.shape == met_times.shape
    np.testing.assert_array_equal(np.isnan(inst_phase), expected_nan_mask)
    assert np.logical_and(
        0 <= inst_phase[~expected_nan_mask], inst_phase[~expected_nan_mask] < 1
    ).all()
