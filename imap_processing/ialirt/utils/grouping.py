"""Common grouping functions for I-ALiRT instruments."""

import numpy as np
import xarray as xr


def filter_valid_groups(grouped_data: xr.Dataset) -> xr.Dataset:
    """
    Filter out groups where `src_seq_ctr` diff are not 1 or -16383.

    Parameters
    ----------
    grouped_data : xr.Dataset
        Dataset with a "group" coordinate.

    Returns
    -------
    filtered_data : xr.Dataset
        Filtered dataset with only valid groups remaining.
    """
    valid_groups = []
    unique_groups = np.unique(grouped_data["group"].values)

    for group in unique_groups:
        src_seq_ctr = grouped_data["src_seq_ctr"][
            (grouped_data["group"] == group).values
        ]
        src_seq_ctr_diff = np.diff(src_seq_ctr)

        # Accept group only if all diffs are 1 or -16383
        if np.all(np.isin(src_seq_ctr_diff, [1, -16383])):
            valid_groups.append(group)

    filtered_data = grouped_data.where(
        xr.DataArray(np.isin(grouped_data["group"], valid_groups), dims="epoch"),
        drop=True,
    )

    return filtered_data


def find_groups(
    accumulated_data: xr.Dataset,
    sequence_range: tuple,
    sequence_name: str,
    time_name: str,
) -> xr.Dataset:
    """
    Group data based on time and sequence number values.

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.
    sequence_range : tuple
        Tuple of two integers defining the range of group values.
    sequence_name : str
        Name of the sequence variable.
    time_name : str
        Name of the time variable.

    Returns
    -------
    grouped_data : xr.Dataset
        Add "group" coordinate.
    """
    sorted_data = accumulated_data.sortby(time_name, ascending=True)

    # Use sequence_range == 0 to define the beginning of the group.
    # Find time at this index and use it as the beginning time for the group.
    start_times = sorted_data[time_name][
        (sorted_data[sequence_name] == sequence_range[0])
    ]
    start_time = start_times.min()
    # Use max sequence_range to define the end of the group.
    end_times = sorted_data[time_name][
        ([sorted_data[sequence_name] == sequence_range[-1]][-1])
    ]
    end_time = end_times.max()

    # Filter data before the sequence_range=0
    # and after the last value of sequence_range.
    grouped_data = sorted_data.where(
        (sorted_data[time_name] >= start_time) & (sorted_data[time_name] <= end_time),
        drop=True,
    )

    # Assign labels based on the start_times.
    group_labels = np.searchsorted(start_times, grouped_data[time_name], side="right")
    # Example:
    # grouped_data.coords
    # Coordinates:
    #   * epoch    (epoch) int64 7kB 315922822184000000 ... 315923721184000000
    #     group    (epoch) int64 7kB 1 1 1 1 1 1 1 1 1 ... 15 15 15 15 15 15 15 15 15
    grouped_data = grouped_data.assign_coords(group=("epoch", group_labels))

    # Filter out groups with non-sequential src_seq_ctr values.
    filtered_data = filter_valid_groups(grouped_data)

    return filtered_data
