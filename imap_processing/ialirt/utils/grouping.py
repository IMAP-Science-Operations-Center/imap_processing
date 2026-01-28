"""Common grouping functions for I-ALiRT instruments."""

import logging

import numpy as np
import xarray as xr

from imap_processing.spice.time import met_to_ttj2000ns, met_to_utc

logger = logging.getLogger(__name__)


def filter_valid_groups(grouped_data: xr.Dataset) -> xr.Dataset:
    """
    Filter out groups where `src_seq_ctr` diff are not 1.

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
        src_seq_ctr_diff = np.diff(src_seq_ctr) % 16384

        # Accept group only if all diffs are 1.
        if np.all(src_seq_ctr_diff == 1):
            valid_groups.append(group)
        else:
            logger.info(f"src_seq_ctr_diff != 1 for group {group}.")

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
    check_src_seq_ctr: bool = True,
) -> xr.Dataset:
    """
    Group data based on time and sequence number values.

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.
    sequence_range : tuple
        Tuple of two integers defining the range of group values (inclusive endpoints).
    sequence_name : str
        Name of the sequence variable.
    time_name : str
        Name of the time variable.
    check_src_seq_ctr : bool | True
        Check for incrementing src_seq_ctr.

    Returns
    -------
    grouped_data : xr.Dataset
        Filtered data with "group" coordinate.

    Notes
    -----
    Filters data based on:
    1. Time values between the first and last sequence_range values.
    Take out time values before sequence_range[0] and after sequence_range[-1].
    2. Sequence values src_seq_ctr between the first and
    last sequence_range. These must be consecutive.
    """
    sorted_data = accumulated_data.sortby(time_name, ascending=True)

    # Use sequence_range == 0 to define the beginning of the group.
    # Find time at this index and use it as the beginning time for the group.
    start_times = sorted_data[time_name][
        (sorted_data[sequence_name] == sequence_range[0])
    ]
    # Use max sequence_range to define the end of the group.
    end_times = sorted_data[time_name][
        ([sorted_data[sequence_name] == sequence_range[-1]][-1])
    ]
    # If no matching start or end times, return empty dataset
    if start_times.size == 0 or end_times.size == 0:
        empty = sorted_data.isel(epoch=[])
        empty = empty.assign_coords(group=("epoch", np.empty(0, dtype=int)))
        return empty

    start_time = start_times.min()
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

    if check_src_seq_ctr:
        # Filter out groups with non-sequential src_seq_ctr values.
        filtered_data = filter_valid_groups(grouped_data)
    else:
        filtered_data = grouped_data

    return filtered_data


def _populate_instrument_header_items(met: np.ndarray) -> dict:
    """
    Create header values.

    Parameters
    ----------
    met : np.ndarray
        Mission elapsed time.

    Returns
    -------
    header : dict
        Header for each instrument.
    """
    sc_met = (met[0] + met[-1]) // 2
    header = {
        "apid": 478,
        "met": int(sc_met),
        "met_in_utc": met_to_utc(sc_met).split(".")[0],
        "ttj2000ns": int(met_to_ttj2000ns(sc_met)),
    }
    return header
