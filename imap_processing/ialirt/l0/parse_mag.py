"""Functions to support MAG processing."""

import logging

import numpy as np
import xarray as xr
from dataclasses import astuple

from imap_processing.ialirt.l0.mag_l0_ialirt_data import decode_packet0, decode_packet1, decode_packet2, decode_packet3

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


def get_science_data(status_values, pkt_counter):
    """Get the science data."""

    packet_0 = decode_packet0(int(status_values[pkt_counter == 0]))
    packet_1 = decode_packet1(int(status_values[pkt_counter == 1]))
    packet_2 = decode_packet2(int(status_values[pkt_counter == 2]))
    packet_3 = decode_packet3(int(status_values[pkt_counter == 3]))
    print('hi')


    return packet


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
    unique_acq_times, group_labels = np.unique(data["mag_acq_tm_coarse"],
                                               return_inverse=True)

    # Assign group labels as a coordinate
    data["group"] = ("group", group_labels)

    return data


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

        science_data = get_science_data(status_values, pkt_counter)

    # Concatenate the packets

    return science_data



