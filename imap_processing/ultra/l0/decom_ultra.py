"""Decommutates Ultra CCSDS packets."""

import logging
from typing import Optional

from imap_processing import decom
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.ultra.l0.decom_tools import (
    decompress_binary,
    decompress_image,
    read_image_raw_events_binary,
)
from imap_processing.ultra.l0.ultra_utils import ParserHelper, UltraParams
from imap_processing.utils import group_by_apid, sort_by_time

logging.basicConfig(level=logging.INFO)


def append_params(
    decom_data: dict, packet, decompressed_data=None, decompressed_key=None
):
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
    decompressed_key : str
        Key for decompressed data.
    """
    parser_helper = ParserHelper()

    for key, item in packet.data.items():
        # Initialize the list for the first time
        if key not in decom_data:
            decom_data[key] = []
        if key != decompressed_key:
            decom_data[key].append(item.derived_value)
        else:
            decom_data[key].append(decompressed_data)

    ccsds_data = CcsdsData(packet.header)
    parser_helper.append_ccsds_fields(decom_data, ccsds_data)


def decom_ultra_apids(packet_file: str, xtce: str, test_apid: Optional[int] = None):
    """
    Unpack and decode Ultra packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.
    test_apid : Optional[int]
        The APID to test. If None, all APIDs are processed.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    packets = decom.decom_packets(packet_file, xtce)
    grouped_data = group_by_apid(packets)
    decom_data = {}

    if test_apid:
        grouped_data = {test_apid: grouped_data[test_apid]}

    for apid in grouped_data.keys():
        if not any(
            apid in category.value.apid
            for category in [
                UltraParams.ULTRA_EVENTS,
                UltraParams.ULTRA_AUX,
                UltraParams.ULTRA_TOF,
                UltraParams.ULTRA_RATES,
            ]
        ):
            logging.info(f"{apid} is currently not supported")
            continue

        sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

        for packet in sorted_packets:
            # Here there are multiple images in a single packet,
            # so we need to loop through each image and decompress it.
            if apid in UltraParams.ULTRA_EVENTS.value.apid:
                decom_data = read_image_raw_events_binary(packet, decom_data)
                count = packet.data["COUNT"].derived_value

                if count == 0:
                    append_params(decom_data, packet)
                else:
                    for i in range(count):
                        logging.info(f"Appending image #{i}")
                        append_params(decom_data, packet)

            elif apid in UltraParams.ULTRA_AUX.value.apid:
                append_params(decom_data, packet)

            elif apid in UltraParams.ULTRA_TOF.value.apid:
                decompressed_data = decompress_image(
                    packet.data["P00"].derived_value,
                    packet.data["PACKETDATA"].raw_value,
                    UltraParams.ULTRA_TOF.value.width,
                    UltraParams.ULTRA_TOF.value.mantissa_bit_length,
                )

                append_params(
                    decom_data,
                    packet,
                    decompressed_data=decompressed_data,
                    decompressed_key="PACKETDATA",
                )

            elif apid in UltraParams.ULTRA_RATES.value.apid:
                decompressed_data = decompress_binary(
                    packet.data["FASTDATA_00"].raw_value,
                    UltraParams.ULTRA_RATES.value.width,
                    UltraParams.ULTRA_RATES.value.block,
                    UltraParams.ULTRA_RATES.value.len_array,
                    UltraParams.ULTRA_RATES.value.mantissa_bit_length,
                )

                append_params(
                    decom_data,
                    packet,
                    decompressed_data=decompressed_data,
                    decompressed_key="FASTDATA_00",
                )

    return decom_data
