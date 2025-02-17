"""Decommutates Ultra CCSDS packets."""

import collections
import logging
from collections import defaultdict
from typing import Any, Union

import numpy as np
from space_packet_parser import packets

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.ultra.l0.decom_tools import (
    decompress_binary,
    decompress_image,
    read_image_raw_events_binary,
)
from imap_processing.ultra.l0.ultra_utils import (
    RATES_KEYS,
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
    append_ccsds_fields,
)
from imap_processing.utils import convert_to_binary_string, sort_by_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def append_tof_params(
    decom_data: dict,
    packet: packets.CCSDSPacket,
    decompressed_data: np.ndarray,
    data_dict: dict,
    stacked_dict: dict,
) -> None:
    """
    Append parsed items to a dictionary, including decompressed data if available.

    Parameters
    ----------
    decom_data : dict
        Dictionary to which the data is appended.
    packet : space_packet_parser.packets.CCSDSPacket
        Individual packet.
    decompressed_data : list
        Data that has been decompressed.
    data_dict : dict
        Dictionary used for stacking in SID dimension.
    stacked_dict : dict
        Dictionary used for stacking in time dimension.
    """
    # TODO: add error handling to make certain every timestamp has 8 SID values

    for key in packet.user_data.keys():
        # Keep appending packet data until SID = 7
        if key == "PACKETDATA":
            data_dict[key].append(decompressed_data)
        # Keep appending all other data until SID = 7
        else:
            data_dict[key].append(packet[key])

    # Append CCSDS fields to the dictionary
    ccsds_data = CcsdsData(packet.header)
    append_ccsds_fields(data_dict, ccsds_data)

    # Once "SID" reaches 7, we have all the images and data for the single timestamp
    if packet["SID"] == 7:
        decom_data["SHCOARSE"].extend(list(set(data_dict["SHCOARSE"])))
        data_dict["SHCOARSE"].clear()

        for key in packet.user_data.keys():
            if key != "SHCOARSE":
                stacked_dict[key].append(np.stack(data_dict[key]))
                data_dict[key].clear()
        for key in packet.header.keys():
            stacked_dict[key].append(np.stack(data_dict[key]))
            data_dict[key].clear()


def append_params(decom_data: dict, packet: packets.CCSDSPacket) -> None:
    # Todo Update what packet type is.
    """
    Append parsed items to a dictionary, including decompressed data if available.

    Parameters
    ----------
    decom_data : dict
        Dictionary to which the data is appended.
    packet : space_packet_parser.packets.CCSDSPacket
        Individual packet.
    """
    for key, value in packet.user_data.items():
        decom_data[key].append(value)

    ccsds_data = CcsdsData(packet.header)
    append_ccsds_fields(decom_data, ccsds_data)


def process_ultra_apids(data: list, apid: int) -> Union[dict[Any, Any], bool]:
    """
    Unpack and decode Ultra packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    data : list
        Grouped data.
    apid : int
        The APID to process.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    # Strategy dict maps APIDs to their respective processing functions
    strategy_dict = {
        ULTRA_TOF.apid[0]: process_ultra_tof,
        ULTRA_EVENTS.apid[0]: process_ultra_events,
        ULTRA_AUX.apid[0]: process_ultra_aux,
        ULTRA_RATES.apid[0]: process_ultra_rates,
    }

    sorted_packets = sort_by_time(data, "SHCOARSE")

    process_function = strategy_dict.get(apid, lambda *args: False)
    decom_data = process_function(sorted_packets, defaultdict(list))

    return decom_data


def process_ultra_tof(
    sorted_packets: list, decom_data: collections.defaultdict
) -> dict:
    """
    Unpack and decode Ultra TOF packets.

    Parameters
    ----------
    sorted_packets : list
        TOF packets sorted by time.
    decom_data : collections.defaultdict
        Empty dictionary.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    stacked_dict: dict = defaultdict(list)
    data_dict: dict = defaultdict(list)

    # For TOF we need to sort by time and then SID
    sorted_packets = sorted(
        sorted_packets,
        key=lambda x: (x["SHCOARSE"].raw_value, x["SID"].raw_value),
    )
    if isinstance(ULTRA_TOF.mantissa_bit_length, int) and isinstance(
        ULTRA_TOF.width, int
    ):
        for packet in sorted_packets:
            binary_data = convert_to_binary_string(packet["PACKETDATA"])
            # Decompress the image data
            decompressed_data = decompress_image(
                packet["P00"],
                binary_data,
                ULTRA_TOF.width,
                ULTRA_TOF.mantissa_bit_length,
            )

            # Append the decompressed data and other derived data
            # to the dictionary
            append_tof_params(
                decom_data,
                packet,
                decompressed_data=decompressed_data,
                data_dict=data_dict,
                stacked_dict=stacked_dict,
            )

    # Stack the data to create required dimensions
    for key in stacked_dict.keys():
        decom_data[key] = np.stack(stacked_dict[key])

    return decom_data


def process_ultra_events(sorted_packets: list, decom_data: dict) -> dict:
    """
    Unpack and decode Ultra EVENTS packets.

    Parameters
    ----------
    sorted_packets : list
        EVENTS packets sorted by time.
    decom_data : collections.defaultdict
        Empty dictionary.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    for packet in sorted_packets:
        # Here there are multiple images in a single packet,
        # so we need to loop through each image and decompress it.
        decom_data = read_image_raw_events_binary(packet, decom_data)
        count = packet["COUNT"]

        if count == 0:
            append_params(decom_data, packet)
        else:
            for i in range(count):
                logging.info(f"Appending image #{i}")
                append_params(decom_data, packet)

    return decom_data


def process_ultra_aux(sorted_packets: list, decom_data: dict) -> dict:
    """
    Unpack and decode Ultra AUX packets.

    Parameters
    ----------
    sorted_packets : list
        AUX packets sorted by time.
    decom_data : collections.defaultdict
        Empty dictionary.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    for packet in sorted_packets:
        append_params(decom_data, packet)

    return decom_data


def process_ultra_rates(sorted_packets: list, decom_data: dict) -> dict:
    """
    Unpack and decode Ultra RATES packets.

    Parameters
    ----------
    sorted_packets : list
        RATES packets sorted by time.
    decom_data : collections.defaultdict
        Empty dictionary.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    if (
        isinstance(ULTRA_RATES.mantissa_bit_length, int)
        and isinstance(ULTRA_RATES.len_array, int)
        and isinstance(ULTRA_RATES.block, int)
        and isinstance(ULTRA_RATES.width, int)
    ):
        for packet in sorted_packets:
            raw_binary_string = convert_to_binary_string(packet["FASTDATA_00"])
            decompressed_data = decompress_binary(
                raw_binary_string,
                ULTRA_RATES.width,
                ULTRA_RATES.block,
                ULTRA_RATES.len_array,
                ULTRA_RATES.mantissa_bit_length,
            )

            for index in range(ULTRA_RATES.len_array):
                decom_data[RATES_KEYS[index]].append(decompressed_data[index])

            append_params(decom_data, packet)

    return decom_data
