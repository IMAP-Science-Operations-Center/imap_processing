"""Functions to support I-ALiRT SWE packet parsing."""

import logging

import numpy as np
import xarray as xr

from imap_processing.swe.l1a.swe_science import decompressed_counts

logger = logging.getLogger(__name__)


def calculate_time(coarse_time: xr.DataArray, fin_time: xr.DataArray) -> xr.DataArray:
    """
    Calculate the time.

    Parameters
    ----------
    coarse_time : xr.DataArray
        Coarse time.
    fin_time : xr.DataArray
        Fine time.

    Returns
    -------
    time_seconds: xr.DataArray
        Calculated time.

    Notes
    -----
    1000000 fine time units = 1 second
    """
    # TODO: Confirm this is correction.
    fine_time_fraction = fin_time / 1000000
    time_seconds = coarse_time + fine_time_fraction

    return time_seconds


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


def find_groups(accumulated_data: xr.Dataset) -> xr.Dataset:
    """
    Group data based on swe_acq_sec and swe_acq_sub values.

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.

    Returns
    -------
    grouped_data : xr.Dataset
        Add "group" coordinate.
    """
    subcom_range = (0, 59)

    time_seconds = calculate_time(
        accumulated_data["swe_acq_sec"], accumulated_data["swe_acq_sub"]
    )
    accumulated_data["time_seconds"] = time_seconds
    sorted_data = accumulated_data.sortby("time_seconds", ascending=True)

    # Use subcom_range == 0 to define the beginning of the group.
    # Find time at this index and use it as the beginning time for the group.
    start_times = sorted_data["time_seconds"][
        (sorted_data["swe_seq"] == subcom_range[0])
    ]
    start_time = start_times.min()
    # Use subcom_range == 59 to define the end of the group.
    end_times = sorted_data["time_seconds"][
        ([sorted_data["swe_seq"] == subcom_range[-1]][-1])
    ]
    end_time = end_times.max()

    # Filter out data before the subcom_range=0 and after the last subcom_range=59.
    grouped_data = sorted_data.where(
        (sorted_data["time_seconds"] >= start_time)
        & (sorted_data["time_seconds"] <= end_time),
        drop=True,
    )

    # Assign labels based on the start_times.
    group_labels = np.searchsorted(
        start_times, grouped_data["time_seconds"], side="right"
    )
    # Example:
    # grouped_data.coords
    # Coordinates:
    #   * epoch    (epoch) int64 7kB 315922822184000000 ... 315923721184000000
    #   * group    (group) int64 7kB 1 1 1 1 1 1 1 1 1 ... 15 15 15 15 15 15 15 15 15
    grouped_data["group"] = ("group", group_labels)

    # Filter out groups with non-sequential src_seq_ctr values.
    filtered_data = filter_valid_groups(grouped_data)

    return filtered_data


def process_swe(accumulated_data: xr.Dataset) -> list[dict]:
    """
    Process SWE.

    Parameters
    ----------
    accumulated_data : xr.Dataset
        Packets dataset accumulated over 1 min.

    Returns
    -------
    swe_data : list[dict]
        Dictionaries of the parsed data product.
    """
    logger.info("Processing SWE.")

    grouped_data = find_groups(accumulated_data)
    unique_groups = np.unique(grouped_data["group"])
    swe_data = []

    for group in unique_groups:
        # Sequence values for the group should be 0-59 with no duplicates.
        seq_values = grouped_data["swe_seq"][(grouped_data["group"] == group).values]

        # Ensure no duplicates and all values from 0 to 59 are present
        if not np.array_equal(seq_values, np.arange(60)):
            logger.warning(
                f"Group {group} does not contain all values from 0 to "
                f"59 without duplicates."
            )
            continue

        # Get science values for each group.
        science_values = grouped_data["mag_data"][
            (grouped_data["group"] == group).values
        ]

        # We know we can only have 8 bit numbers input, so iterate over all
        # possibilities once up front
        decompression_table = np.array([decompressed_counts(i) for i in range(256)])

        # Loop through each packet individually with a list comprehension and
        # perform the following steps:
        # 1. Turn the binary string  of 0s and 1s to an int
        # 2. Convert the int into a bytes object of length 1260 (10080 / 8)
        #    Eg. "0000000011110011" --> b'\x00\xf3'
        #    1260 = 15 seconds x 12 energy steps x 7 CEMs
        # 3. Read that bytes data to a numpy array of uint8 through the buffer protocol
        # 4. Reshape the data to 180 x 7
        raw_science_array = np.array(
            [
                np.frombuffer(binary_string, dtype=np.uint8).reshape(
                    180, swe_constants.N_CEMS
                )
                for binary_string in l0_dataset["science_data"].values
            ]
        )

        # Decompress the raw science data using numpy broadcasting logic
        # science_array will be the same shape as raw_science_array (npackets, 180, 7)
        science_array = decompression_table[raw_science_array]

    return swe_data
