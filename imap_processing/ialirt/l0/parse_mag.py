"""Functions to support I-ALiRT MAG packet parsing."""

import logging

import numpy as np
import xarray as xr

from imap_processing.ialirt.l0.mag_l0_ialirt_data import (
    decode_packet0,
    decode_packet1,
    decode_packet2,
    decode_packet3,
)

logger = logging.getLogger(__name__)


def get_pkt_counter(status_values: xr.DataArray) -> xr.DataArray:
    """
    Get the packet counters.

    Parameters
    ----------
    status_values : xr.DataArray
        Status data.

    Returns
    -------
    pkt_counters : xr.DataArray
        Packet counters.
    """
    # mag_status is a 24 bit unsigned field
    # The leading 2 bits of STATUS are a 2 bit 0-3 counter
    pkt_counter = (status_values >> 22) & 0x03

    return pkt_counter


def get_status_data(status_values: xr.DataArray, pkt_counters: xr.DataArray) -> dict:
    """
    Get the status data.

    Parameters
    ----------
    status_values : xr.DataArray
        Status data.
    pkt_counters : xr.DataArray
        Packet counters.

    Returns
    -------
    combined_packets : dict
        Decoded packets.
    """
    decoders = {
        0: decode_packet0,
        1: decode_packet1,
        2: decode_packet2,
        3: decode_packet3,
    }

    combined_packets = {}

    for pkt_num, decoder in decoders.items():
        status_subset = status_values[pkt_counters == pkt_num]
        decoded_packet = decoder(int(status_subset))
        combined_packets.update(vars(decoded_packet))

    return combined_packets


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
    65535 fine time units = 1 second
    """
    fine_time_fraction = fin_time / 65535.0
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


def find_groups(data: xr.Dataset) -> xr.Dataset:
    """
    Group data based on `mag_acq_tm_coarse` values.

    Parameters
    ----------
    data : xr.Dataset
        Packets dataset.

    Returns
    -------
    grouped_data : xr.Dataset
        Grouped data with an additional "group" coordinate.
    """
    pkt_range = (0, 3)

    time_seconds = calculate_time(data["mag_acq_tm_coarse"], data["mag_acq_tm_fine"])
    data["time_seconds"] = time_seconds
    sorted_data = data.sortby("time_seconds", ascending=True)
    status_values = sorted_data["mag_status"]

    pkt_counter = get_pkt_counter(status_values)
    data["pkt_counter"] = pkt_counter

    # Use pkt_counter == 0 to define the beginning of the group.
    # Find time at this index and use it as the beginning time for the group.
    start_times = data["time_seconds"][(pkt_counter == pkt_range[0])]
    start_time = start_times.min()
    # Use pkt_counter == 3 to define the end of the group.
    end_times = data["time_seconds"][([pkt_counter == pkt_range[-1]][-1])]
    end_time = end_times.max()

    # Filter out data before the pkt_counter=0 and after the last pkt_counter=3.
    grouped_data = data.where(
        (data["time_seconds"] >= start_time) & (data["time_seconds"] <= end_time),
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


def get_bytes(val: int) -> list[int]:
    """
    Extract three bytes from a 24-bit integer.

    Parameters
    ----------
    val : int
        24-bit integer value.

    Returns
    -------
    list[int]
        List of three extracted bytes.
    """
    return [
        (val >> 16) & 0xFF,  # Most significant byte (Byte2)
        (val >> 8) & 0xFF,  # Middle byte (Byte1)
        (val >> 0) & 0xFF,  # Least significant byte (Byte0)
    ]


def extract_magnetic_vectors(science_values: xr.DataArray) -> dict:
    """
    Extract the magnetic vectors.

    Parameters
    ----------
    science_values : xr.DataArray
        Science data.

    Returns
    -------
    vectors : dict
        Magnetic vectors.
    """
    # Convert each 24-bit value to its three constituent bytes
    science0 = get_bytes(int(science_values[0]))
    science1 = get_bytes(int(science_values[1]))
    science2 = get_bytes(int(science_values[2]))
    science3 = get_bytes(int(science_values[3]))

    # Primary sensor:
    pri_x = (science0[0] << 8) | science0[1]
    pri_y = (science0[2] << 8) | science1[0]
    pri_z = (science1[1] << 8) | science1[2]

    # Secondary sensor:
    sec_x = (science2[0] << 8) | science2[1]
    sec_y = (science2[2] << 8) | science3[0]
    sec_z = (science3[1] << 8) | science3[2]

    vectors = {
        "pri_x": pri_x,
        "pri_y": pri_y,
        "pri_z": pri_z,
        "sec_x": sec_x,
        "sec_y": sec_y,
        "sec_z": sec_z,
    }

    return vectors


def get_time(grouped_data: xr.Dataset, group: int, pkt_counter: xr.DataArray) -> dict:
    """
    Get the time for the grouped data.

    Parameters
    ----------
    grouped_data : xr.Dataset
        Grouped data.
    group : int
        Group number.
    pkt_counter : xr.DataArray
        Packet counter.

    Returns
    -------
    time_data : dict
        Coarse and fine time for Primary and Secondary Sensors.
    """
    pri_coarsetm = grouped_data["mag_acq_tm_coarse"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 0]

    pri_fintm = grouped_data["mag_acq_tm_fine"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 0]

    sec_coarsetm = grouped_data["mag_acq_tm_coarse"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 2]

    sec_fintm = grouped_data["mag_acq_tm_fine"][
        (grouped_data["group"] == group).values
    ][pkt_counter == 2]

    time_data = {
        "pri_coarsetm": int(pri_coarsetm),
        "pri_fintm": int(pri_fintm),
        "sec_coarsetm": int(sec_coarsetm),
        "sec_fintm": int(sec_fintm),
    }

    return time_data


def parse_packet(xarray_data: xr.Dataset) -> list[dict]:
    """
    Parse the MAG packets.

    Parameters
    ----------
    xarray_data : xr.Dataset
        Packet data.

    Returns
    -------
    mag_data : list[dict]
        Dictionaries of the parsed data product.
    """
    logger.info("Parsing MAG.")

    grouped_data = find_groups(xarray_data)
    unique_groups = np.unique(grouped_data["group"])
    mag_data = []

    for group in unique_groups:
        # Get status values for each group.
        status_values = grouped_data["mag_status"][
            (grouped_data["group"] == group).values
        ]
        pkt_counter = grouped_data["pkt_counter"][
            (grouped_data["group"] == group).values
        ]

        if not np.array_equal(pkt_counter, np.arange(4)):
            logger.warning(
                f"Group {group} does not contain all values from 0 to "
                f"3 without duplicates."
            )
            continue

        # Get decoded status data.
        status_data = get_status_data(status_values, pkt_counter)

        # Get science values for each group.
        science_values = grouped_data["mag_data"][
            (grouped_data["group"] == group).values
        ]
        science_data = extract_magnetic_vectors(science_values)

        # Get time values for each group.
        time_data = get_time(grouped_data, group, pkt_counter)

        mag_data.append({**status_data, **science_data, **time_data})

    return mag_data
