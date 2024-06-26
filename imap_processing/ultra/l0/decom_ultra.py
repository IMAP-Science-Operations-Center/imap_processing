"""Decommutates Ultra CCSDS packets."""

import collections
import logging
from collections import defaultdict

import numpy as np

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
from imap_processing.utils import sort_by_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def append_tof_params(  # type: ignore[no-untyped-def] #Todo Update, need packet param type.
    decom_data: dict,
    packet,
    decompressed_data: list,
    data_dict: dict,
    stacked_dict: dict,
) -> None:
    """
    Append parsed items to a dictionary, including decompressed data if available.

    Parameters
    ----------
    decom_data : dict
        Dictionary to which the data is appended.
    packet : space_packet_parser.parser.Packet
        Individual packet.
    decompressed_data : list
        Data that has been decompressed.
    data_dict : dict
        Dictionary used for stacking in SID dimension.
    stacked_dict : dict
        Dictionary used for stacking in time dimension.
    """
    # TODO: add error handling to make certain every timestamp has 8 SID values

    for key in packet.data.keys():
        # Keep appending packet data until SID = 7
        if key == "PACKETDATA":
            data_dict[key].append(decompressed_data)
        # Keep appending all other data until SID = 7
        else:
            data_dict[key].append(packet.data[key].derived_value)

    # Append CCSDS fields to the dictionary
    ccsds_data = CcsdsData(packet.header)
    append_ccsds_fields(data_dict, ccsds_data)

    # Once "SID" reaches 7, we have all the images and data for the single timestamp
    if packet.data["SID"].derived_value == 7:
        decom_data["SHCOARSE"].extend(list(set(data_dict["SHCOARSE"])))
        data_dict["SHCOARSE"].clear()

        for key in packet.data.keys():
            if key != "SHCOARSE":
                stacked_dict[key].append(np.stack(data_dict[key]))
                data_dict[key].clear()
        for key in packet.header.keys():
            stacked_dict[key].append(np.stack(data_dict[key]))
            data_dict[key].clear()


def append_params(decom_data: dict, packet) -> None:  # type: ignore[no-untyped-def]
    # Todo Update what packet type is.
    """
    Append parsed items to a dictionary, including decompressed data if available.

    Parameters
    ----------
    decom_data : dict
        Dictionary to which the data is appended.
    packet : space_packet_parser.parser.Packet
        Individual packet.
    """
    for key, item in packet.data.items():
        decom_data[key].append(item.derived_value)

    ccsds_data = CcsdsData(packet.header)
    append_ccsds_fields(decom_data, ccsds_data)


def process_ultra_apids(data: list, apid: int) -> dict:
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

    process_function = strategy_dict.get(apid)
    decom_data = process_function(sorted_packets, defaultdict(list))  # type: ignore[misc]
    # ToDo Change, "None" not callable

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
        key=lambda x: (x.data["SHCOARSE"].raw_value, x.data["SID"].raw_value),
    )

    for packet in sorted_packets:
        # Decompress the image data
        decompressed_data = decompress_image(
            packet.data["P00"].derived_value,
            packet.data["PACKETDATA"].raw_value,
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
        count = packet.data["COUNT"].derived_value

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
    for packet in sorted_packets:
        decompressed_data = decompress_binary(
            packet.data["FASTDATA_00"].raw_value,
            ULTRA_RATES.width,
            ULTRA_RATES.block,
            ULTRA_RATES.len_array,
            ULTRA_RATES.mantissa_bit_length,
        )

        for index in range(ULTRA_RATES.len_array):  # type: ignore[arg-type]
            # TODO Change, Argument 1 to "range" has incompatible type
            # "Optional[int]"; expected "SupportsIndex"
            decom_data[RATES_KEYS[index]].append(decompressed_data[index])

        append_params(decom_data, packet)

    return decom_data
