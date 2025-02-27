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
    # Ensure data is sorted by `mag_acq_tm_coarse`
    data = data.sortby("mag_acq_tm_coarse", ascending=True)

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
        "PRI_X": pri_x,
        "PRI_Y": pri_y,
        "PRI_Z": pri_z,
        "SEC_X": sec_x,
        "SEC_Y": sec_y,
        "SEC_Z": sec_z,
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
        "PRI_COARSETM": int(pri_coarsetm),
        "PRI_FINTM": int(pri_fintm),
        "SEC_COARSETM": int(sec_coarsetm),
        "SEC_FINTM": int(sec_fintm),
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
    logger.info("Calculating DE.")

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
