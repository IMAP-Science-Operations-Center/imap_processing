"""Functions to support I-ALiRT MAG packet parsing."""

import logging

import numpy as np
from numpy.typing import NDArray
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


def unwrap_src_seq_ctr(src_seq_ctr: NDArray, mag_acq_tm_coarse: NDArray,
                       pkt_counter: NDArray) -> NDArray:
    """
    Unwrap a 14-bit src_seq_ctr to handle counter rollovers.

    The counter wraps at max_seq - 1 (default 16383 for 14-bit counters),
    so the unwrapped version will be strictly increasing across rollovers.

    Parameters
    ----------
    src_seq_ctr : NDArray
        Source sequence counter values.

    Returns
    -------
    unwrapped_seq : NDArray
        Unwrapped sequence counter.
    """
    unwrapped_seq = src_seq_ctr.copy()
    max_seq = 16384  # 2^14, max value + 1 for 14-bit counter

    # Detect where the counter wraps
    rollovers = np.diff(src_seq_ctr) < 0

    # Create an array that increments by 1 at each rollover
    rollover_count = np.zeros_like(src_seq_ctr, dtype=int)
    rollover_count[1:] = np.cumsum(rollovers)

    # Apply the unwrapping adjustment
    unwrapped_seq += rollover_count * max_seq

    return unwrapped_seq


def sort_by_unwrapped_seq_and_pkt_counter(dataset: xr.Dataset, pkt_counter: xr.DataArray,
                                          src_seq_ctr_name="src_seq_ctr") -> xr.Dataset:
    """
    Sort an xarray Dataset by unwrapped src_seq_ctr and pkt_counter.

    Handles 14-bit counter wraparound (0-16383) by unwrapping src_seq_ctr
    into a strictly increasing sequence.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to sort.
    pkt_counter : xr.DataArray
        Packet counter (same length as dataset's epoch dimension).
    src_seq_ctr_name : str, optional
        Name of the src_seq_ctr variable in the dataset.

    Returns
    -------
    xr.Dataset
        Sorted dataset.
    """
    MAX_SEQ = 16384  # 2^14, max value + 1 for 14-bit counter

    # Extract src_seq_ctr from dataset
    src_seq_ctr = dataset[src_seq_ctr_name].values

    # Initialize unwrapped sequence counter
    unwrapped_seq = src_seq_ctr.copy()
    rollover_count = 0

    # Unwrap the sequence counter
    for i in range(1, len(src_seq_ctr)):
        if src_seq_ctr[i] < src_seq_ctr[i - 1]:
            rollover_count += 1
        unwrapped_seq[i] += rollover_count * MAX_SEQ

    # Combine unwrapped sequence counter with pkt_counter into a sortable key
    combined_key = list(zip(unwrapped_seq, pkt_counter.values))

    # Sort indices based on combined key
    sort_indices = np.argsort(combined_key)

    # Apply sorting to the dataset
    sorted_dataset = dataset.isel(epoch=sort_indices)

    return sorted_dataset


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

    data = data.sortby("mag_acq_tm_coarse", ascending=True)
    status_values = data["mag_status"]

    pkt_counter = get_pkt_counter(status_values)

    # pkt_counter == 0 to define the beginning of the group.
    src_seq_ctr = data["src_seq_ctr"]

    # Get unique acquisition times and create group labels
    _, group_labels = np.unique(data["mag_acq_tm_coarse"], return_inverse=True)

    # Assign group labels as a coordinate
    data["group"] = ("group", group_labels)

    return data


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
        # Get the packet counters for each group.
        pkt_counter = get_pkt_counter(status_values)

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
