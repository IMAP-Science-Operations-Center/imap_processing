"""Functions to support MAG processing."""

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


def get_pkt_counter(mag_status):
    """
    Get the packet number.

    Parameters
    ----------
    mag_status : NDArray
        Status data.

    Returns
    -------
    pkt_counter : int
        Packet counter.
    """
    # mag_status is a 24‑bit unsigned field
    # The leading 2 bits of STATUS are a 2 bit 0-3 counter
    pkt_counter = (mag_status >> 22) & 0x03

    return pkt_counter


def get_status_data(status_values, pkt_counter):
    """Get the science data."""
    decoders = {
        0: decode_packet0,
        1: decode_packet1,
        2: decode_packet2,
        3: decode_packet3,
    }

    combined_packets = {}

    for pkt_num, decoder in decoders.items():
        status_subset = status_values[pkt_counter == pkt_num]
        decoded_packet = decoder(int(status_subset))
        combined_packets.update(vars(decoded_packet))

    return combined_packets


def find_groups(data: xr.Dataset) -> xr.Dataset:
    """
    Group data based on `mag_acq_tm_coarse` values.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing `mag_acq_tm_coarse`.

    Returns
    -------
    grouped_data : xr.Dataset
        Grouped data with an additional "group" coordinate.
    """
    # Ensure data is sorted by `mag_acq_tm_coarse`
    data = data.sortby("mag_acq_tm_coarse", ascending=True)

    # Get unique acquisition times and create group labels
    unique_acq_times, group_labels = np.unique(
        data["mag_acq_tm_coarse"], return_inverse=True
    )

    # Assign group labels as a coordinate
    data["group"] = ("group", group_labels)

    return data

import numpy as np

def uint24_to_bytes(uint24_array):
    """
    Convert an array of uint24 values into bytes.
    """
    byte_array = np.zeros((len(uint24_array), 3), dtype=np.uint8)
    byte_array[:, 0] = (uint24_array >> 16) & 0xFF  # Extract first byte
    byte_array[:, 1] = (uint24_array >> 8) & 0xFF   # Extract second byte
    byte_array[:, 2] = (uint24_array >> 0) & 0xFF   # Extract third byte
    return byte_array


import numpy as np


def extract_magnetic_vectors(mag_data: np.ndarray) -> tuple[int, int, int, int, int, int]:

    def get_bytes(val: int) -> list[int]:
        # Extract the three bytes from a 24-bit value (big-endian order)
        return [
            (val >> 16) & 0xFF,  # Most significant byte (Byte2)
            (val >> 8) & 0xFF,  # Middle byte (Byte1)
            (val >> 0) & 0xFF  # Least significant byte (Byte0)
        ]

    # Convert each 24-bit value to its three constituent bytes
    science0 = get_bytes(int(mag_data[0]))
    science1 = get_bytes(int(mag_data[1]))
    science2 = get_bytes(int(mag_data[2]))
    science3 = get_bytes(int(mag_data[3]))

    # Primary sensor:
    # priX: combine first two bytes of Packet 0
    priX = (science0[0] << 8) | science0[1]
    # priY: combine the third byte of Packet 0 (as high byte) with the first byte of Packet 1 (as low byte)
    priY = (science0[2] << 8) | science1[0]
    # priZ: combine the second and third bytes of Packet 1
    priZ = (science1[1] << 8) | science1[2]

    # Secondary sensor:
    # secX: combine the first two bytes of Packet 2
    secX = (science2[0] << 8) | science2[1]
    # secY: combine the third byte of Packet 2 with the first byte of Packet 3
    secY = (science2[2] << 8) | science3[0]
    # secZ: combine the second and third bytes of Packet 3
    secZ = (science3[1] << 8) | science3[2]

    return priX, priY, priZ, secX, secY, secZ


def parse_packet(xarray_data: xr.Dataset):
    """Return science_data in the form of xarray."""
    logger.info("Calculating DE.")

    grouped_data = find_groups(xarray_data)
    unique_groups = np.unique(grouped_data["group"])

    for group in unique_groups:
        status_values = grouped_data["mag_status"][
            (grouped_data["group"] == group).values
        ]
        pkt_counter = get_pkt_counter(status_values)

        if not np.array_equal(pkt_counter, np.arange(4)):
            logger.warning(
                f"Group {group} does not contain all values from 0 to "
                f"3 without duplicates."
            )
            continue

        status_data = get_status_data(status_values, pkt_counter)

        science_values = grouped_data["mag_data"][
            (grouped_data["group"] == group).values
        ]
        priX = extract_magnetic_vectors(science_values)
    # Concatenate the packets

    return status_data
