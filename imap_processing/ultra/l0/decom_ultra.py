"""Decommutates Ultra CCSDS packets."""

import logging

from imap_processing import decom
from imap_processing.ultra.l0.ultra_utils import ParserHelper
from imap_processing.ultra.l0.ultra_utils import UltraParams
from imap_processing.utils import group_by_apid, sort_by_time
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.ultra.l0.decom_tools import decompress_binary, decompress_image, read_image_raw_events_binary

logging.basicConfig(level=logging.INFO)


def append_params(decom_data, packet, excluded_data=None, excluded_key=None):
    """
    Unpack and decode ultra packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    decom_data : dict
        Path to the CCSDS data packet file.
    packet : str
        Path to the XTCE packet definition file.
    test_apid : int
        The APID to test. If None, all APIDs are processed.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    parser_helper = ParserHelper()

    for key, item in packet.data.items():
        # Initialize the list for the first time
        if key not in decom_data:
            decom_data[key] = []
        if key != excluded_key:
            decom_data[key].append(item.derived_value)
        else:
            decom_data[key].append(excluded_data)

    ccsds_data = CcsdsData(packet.header)
    parser_helper.append_ccsds_fields(decom_data, ccsds_data)

def decom_ultra_apids(packet_file: str, xtce: str, test_apid: int = None):
    """
    Unpack and decode ultra packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.
    test_apid : int
        The APID to test. If None, all APIDs are processed.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    packets = decom.decom_packets(packet_file, xtce)
    grouped_data = group_by_apid(packets)
    decom_data = {}
    parser_helper = ParserHelper()

    if test_apid:
        grouped_data = {test_apid: grouped_data[test_apid]}

    for apid in grouped_data.keys():
        sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

        for packet in sorted_packets:
            if apid in UltraParams.ULTRA_EVENTS.value.apid:
                decom_data = read_image_raw_events_binary(packet, decom_data)
                count = packet.data["COUNT"].derived_value

                for i in range(count):
                    ccsds_data = CcsdsData(packet.header)
                    parser_helper.append_ccsds_fields(decom_data, ccsds_data)

            elif apid in UltraParams.ULTRA_AUX.value.apid:

                append_params(decom_data, packet)

            elif apid in UltraParams.ULTRA_TOF.value.apid:

                decompressed_data = decompress_image(
                    packet.data["P00"].derived_value,
                    packet.data["PACKETDATA"].raw_value,
                    UltraParams.ULTRA_TOF
                    .value.width,
                    UltraParams.ULTRA_TOF
                    .value.mantissa_bit_length)

                append_params(decom_data, packet, excluded_data=decompressed_data, excluded_key="PACKETDATA")

            elif apid in UltraParams.ULTRA_RATES.value.apid:

                decompressed_data = decompress_binary(
                    packet.data["FASTDATA_00"].raw_value,
                    UltraParams.ULTRA_RATES.value.width,
                    UltraParams.ULTRA_RATES.value.block,
                    UltraParams.ULTRA_RATES.value.len_array,
                    UltraParams.ULTRA_RATES.value.mantissa_bit_length,
                )

                append_params(decom_data, packet, excluded_data=decompressed_data, excluded_key="FASTDATA_00")

            else:
                logging.info(f"{apid} is currently not supported")

        return decom_data
