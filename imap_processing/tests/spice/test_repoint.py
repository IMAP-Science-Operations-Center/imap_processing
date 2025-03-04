"""Test coverage for imap_processing.spice.repoint.py"""

import numpy as np
import pandas as pd
import pytest

from imap_processing.spice.repoint import get_repoint_data, interpolate_repoint_data


@pytest.fixture()
def fake_repoint_data(monkeypatch, spice_test_data_path):
    """Generate fake spin dataframe for testing"""
    fake_repoint_path = spice_test_data_path / "fake_repoint_data.csv"
    monkeypatch.setenv("REPOINT_DATA_FILEPATH", str(fake_repoint_path))
    return fake_repoint_path


def test_get_repoint_data(fake_repoint_data):
    """Test coverarge for get_repoint_data function."""
    repoint_df = get_repoint_data()
    assert isinstance(repoint_df, pd.DataFrame)
    assert set(repoint_df.columns) == {
        "repoint_start_time",
        "repoint_end_time",
        "repoint_id",
    }


def test_spin_data_no_table():
    """Test coverage for get_repoint_data function when the env var is not set."""
    with pytest.raises(
        ValueError, match="REPOINT_DATA_FILEPATH environment variable is not set."
    ):
        get_repoint_data()


def test_interpolate_repoint_data(fake_repoint_data):
    """Test coverage for get_repoint_data function."""
    query_times = np.array([0, 6, 32])
    expected_vals = {
        "repoint_start_time": np.array([0, 0, 25]),
        "repoint_end_time": np.array([5, 5, 30]),
        "repoint_id": np.array([0, 0, 2]),
        "repoint_in_progress": np.array([True, False, False]),
    }
    repoint_df = interpolate_repoint_data(query_times)

    for key, expected_array in expected_vals.items():
        np.testing.assert_array_equal(repoint_df[key].values, expected_array)


@pytest.mark.parametrize(
    "query_times, match_str",
    [
        (
            np.array([-1, 9]),
            "1 query times are before",
        ),  # Query times before start of table
        (
            np.array([0, 24 * 60 * 60 + 26]),
            "1 query times are after",
        ),  # Query times after end of valid range
    ],
)
def test_interpolate_repoint_data_exceptions(query_times, match_str, fake_repoint_data):
    # Test when query time is
    # Test raising a ValueError when the query time is in between an end and the
    # next start time.
    with pytest.raises(ValueError, match=match_str):
        _ = interpolate_repoint_data(query_times)


def test_interpolate_repoint_data_with_use_fake_fixture(use_fake_repoint_data_for_time):
    """Test coverage for using use_fake_repoint_data_for_time fixutre."""
    repoint_period = 24 * 60 * 60
    repoint_start_times = np.arange(1000, 1000 + 10 * repoint_period, repoint_period)
    _ = use_fake_repoint_data_for_time(repoint_start_times, repoint_id_start=10)
    # Query times are all start times concatenated with 16 minutes after each start time
    query_times = np.concat([repoint_start_times, repoint_start_times + 16 * 60])
    repoint_df = interpolate_repoint_data(query_times)

    # Expected repoint_start_times are the seeded start times array repeated twice
    np.testing.assert_array_equal(
        repoint_df["repoint_start_time"].values,
        np.concat([repoint_start_times, repoint_start_times]),
    )
    # Expected repoint_end_times are the seeded start times + 15 minutes
    np.testing.assert_array_equal(
        repoint_df["repoint_end_time"].values,
        np.concat([repoint_start_times + 15 * 60, repoint_start_times + 15 * 60]),
    )
    # Expected repoint_id
    np.testing.assert_array_equal(
        repoint_df["repoint_id"].values,
        np.concat(
            [
                np.arange(repoint_start_times.size) + 10,
                np.arange(repoint_start_times.size) + 10,
            ]
        ),
    )
    # Expected repoint_in_progress
    np.testing.assert_array_equal(
        repoint_df["repoint_in_progress"].values,
        np.concat(
            [
                np.full(repoint_start_times.size, True),
                np.full(repoint_start_times.size, False),
            ]
        ),
    )
