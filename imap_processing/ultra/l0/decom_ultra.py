"""Decommutates Ultra CCSDS packets."""

import logging
from collections import defaultdict

from imap_processing import decom
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
from imap_processing.utils import group_by_apid, sort_by_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    for key, item in packet.data.items():
        decom_data[key].append(
            decompressed_data if key == decompressed_key else item.derived_value
        )

    ccsds_data = CcsdsData(packet.header)
    append_ccsds_fields(decom_data, ccsds_data)


def decom_ultra_apids(packet_file: str, xtce: str, apid: int):
    """
    Unpack and decode Ultra packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.
    apid : int
        The APID to process.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    packets = decom.decom_packets(packet_file, xtce)
    grouped_data = group_by_apid(packets)
    data = {apid: grouped_data[apid]}

    decom_data = defaultdict(list)

    # Convert decom_data to defaultdict(list) if it's not already
    if not isinstance(decom_data, defaultdict):
        decom_data = defaultdict(list, decom_data)

    for apid in data:
        if not any(
            apid in category.apid
            for category in [
                ULTRA_EVENTS,
                ULTRA_AUX,
                ULTRA_TOF,
                ULTRA_RATES,
            ]
        ):
            logger.info(f"{apid} is currently not supported")
            continue

        sorted_packets = sort_by_time(data[apid], "SHCOARSE")

        for packet in sorted_packets:
            # Here there are multiple images in a single packet,
            # so we need to loop through each image and decompress it.
            if apid in ULTRA_EVENTS.apid:
                decom_data = read_image_raw_events_binary(packet, decom_data)
                count = packet.data["COUNT"].derived_value

                if count == 0:
                    append_params(decom_data, packet)
                else:
                    for i in range(count):
                        logging.info(f"Appending image #{i}")
                        append_params(decom_data, packet)

            elif apid in ULTRA_AUX.apid:
                append_params(decom_data, packet)

            elif apid in ULTRA_TOF.apid:
                decompressed_data = decompress_image(
                    packet.data["P00"].derived_value,
                    packet.data["PACKETDATA"].raw_value,
                    ULTRA_TOF.width,
                    ULTRA_TOF.mantissa_bit_length,
                )

                append_params(
                    decom_data,
                    packet,
                    decompressed_data=decompressed_data,
                    decompressed_key="PACKETDATA",
                )

            elif apid in ULTRA_RATES.apid:
                decompressed_data = decompress_binary(
                    packet.data["FASTDATA_00"].raw_value,
                    ULTRA_RATES.width,
                    ULTRA_RATES.block,
                    ULTRA_RATES.len_array,
                    ULTRA_RATES.mantissa_bit_length,
                )

                for index in range(ULTRA_RATES.len_array):
                    decom_data[RATES_KEYS[index]].append(decompressed_data[index])

                append_params(decom_data, packet)

    return decom_data
