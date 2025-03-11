"""Tests grouping functions for I-ALiRT instruments."""

import numpy as np
import pytest
import xarray as xr

from imap_processing.ialirt.utils.grouping import filter_valid_groups, find_groups


@pytest.fixture()
def test_data():
    """Creates grouped data for filter_valid_groups test."""
    epoch = np.arange(12)

    # Example `src_seq_ctr` values for 3 groups:
    # Group 0 - valid, all diffs = 1
    # Group 1 - invalid, has a jump of 5
    # Group 2 - valid, wraps at -16383
    src_seq_ctr = np.concatenate(
        [
            np.arange(100, 104),
            np.array([200, 205, 206, 207]),
            np.array([16382, 16383, 0, 1]),
        ],
        dtype=np.int32,
    )

    test_data = xr.Dataset(
        data_vars={
            "src_seq_ctr": ("epoch", src_seq_ctr),
            "sequence": ("epoch", np.tile(np.arange(4), 3)),
            "time_seconds": ("epoch", np.arange(12)),
        },
        coords={
            "epoch": epoch,
        },
    )

    return test_data


@pytest.fixture()
def grouped_data():
    """Creates grouped data for filter_valid_groups test."""
    epoch = np.arange(12)

    # Example `src_seq_ctr` values for 3 groups:
    # Group 0 - valid, all diffs = 1
    # Group 1 - invalid, has a jump of 5
    # Group 2 - valid, wraps at -16383
    src_seq_ctr = np.concatenate(
        [
            np.arange(100, 104),
            np.array([200, 205, 206, 207]),
            np.array([16382, 16383, 0, 1]),
        ],
        dtype=np.int32,
    )

    group = np.tile(np.arange(3), 4).reshape(4, 3).T.ravel()

    grouped_data = xr.Dataset(
        data_vars={"src_seq_ctr": ("epoch", src_seq_ctr)},
        coords={"epoch": epoch, "group": ("epoch", group)},
    )

    return grouped_data


def test_filter_valid_groups(grouped_data):
    """Tests filter_valid_groups function."""
    filtered_data = filter_valid_groups(grouped_data)

    assert np.all(np.unique(filtered_data["group"]) == np.array([0, 2]))


def test_find_groups(test_data):
    """Tests the find_groups function."""
    grouped_data = find_groups(test_data, (0, 3), "sequence", "time_seconds")

    assert np.all(np.unique(grouped_data["group"]) == np.array([1, 3]))
