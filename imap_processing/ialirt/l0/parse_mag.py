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


def to_signed_16(n):
    """Convert unsigned 16-bit integer to signed 16-bit integer."""
    n = n & 0xFFFF  # Ensure it's 16-bit
    return n - (0x10000 if n & 0x8000 else 0)


def concatenate_science_data(science_data, pkt_counter):
    """Concatenate the science data for primary and secondary sensors."""

    science0 = int(science_data[pkt_counter == 0])
    science1 = int(science_data[pkt_counter == 1])
    science2 = int(science_data[pkt_counter == 2])
    science3 = int(science_data[pkt_counter == 3])

    # Concatenate values and convert to signed 16-bit integers
    priX = to_signed_16((science0[:, 0] << 8) | science0[:, 1])
    priY = to_signed_16((science0[:, 2] << 8) | science1[:, 0])
    priZ = to_signed_16((science1[:, 1] << 8) | science1[:, 2])

    secX = to_signed_16((science2[:, 0] << 8) | science2[:, 1])
    secY = to_signed_16((science2[:, 2] << 8) | science3[:, 0])
    secZ = to_signed_16((science3[:, 1] << 8) | science3[:, 2])

    return {
        "priX": priX,
        "priY": priY,
        "priZ": priZ,
        "secX": secX,
        "secY": secY,
        "secZ": secZ,
    }


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
        science_data = concatenate_science_data(science_values, pkt_counter)

    # Concatenate the packets

    return status_data
