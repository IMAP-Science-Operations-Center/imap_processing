"""Decommutates Ultra CCSDS packets."""

import logging
from enum import Enum
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import decom
from imap_processing.ultra.l0.ultra_utils import ParserHelper
from imap_processing.ultra.l0.ultra_utils import UltraParams
from imap_processing.utils import group_by_apid, sort_by_time
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.ultra.l0.decom_tools import decompress_binary, decompress_image, read_image_raw_events_binary

logging.basicConfig(level=logging.INFO)


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
    xr.Dataset
        A dataset containing the decoded data fields with 'time' as the coordinating
        dimension.
    """
    packets = decom.decom_packets(packet_file, xtce)
    grouped_data = group_by_apid(packets)
    decom_data = {}
    parser_helper = ParserHelper()

    if test_apid:
        grouped_data = {test_apid: grouped_data[test_apid]}

    for apid in grouped_data.keys():

        if apid in UltraParams.ULTRA_EVENTS.value.apid:
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            for packet in sorted_packets:
                decom_data = read_image_raw_events_binary(packet, decom_data)
                count = packet.data["COUNT"].derived_value

                for i in range(count):
                    ccsds_data = CcsdsData(packet.header)
                    parser_helper.append_ccsds_fields(decom_data, ccsds_data)

        elif apid in UltraParams.ULTRA_AUX.value.apid:
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            for packet in sorted_packets:
                for key, item in packet.data.items():
                    if key not in decom_data:
                        # Initialize the list for the first time
                        decom_data[key] = []
                    decom_data[key].append(item.derived_value)

                ccsds_data = CcsdsData(packet.header)
                parser_helper.append_ccsds_fields(decom_data, ccsds_data)

        elif apid in UltraParams.ULTRA_TOF.value.apid:
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            for packet in sorted_packets:
                decompressed_data = decompress_image(
                    packet.data["P00"].derived_value,
                    packet.data["PACKETDATA"].raw_value,
                    UltraParams.ULTRA_TOF
                    .value.width,
                    UltraParams.ULTRA_TOF
                    .value.mantissa_bit_length)

                for key, item in packet.data.items():
                    # Initialize the list for the first time
                    if key not in decom_data:
                        decom_data[key] = []
                    if key != "PACKETDATA":
                        decom_data[key].append(item.derived_value)
                    else:
                        decom_data[key].append(decompressed_data)

                ccsds_data = CcsdsData(packet.header)
                parser_helper.append_ccsds_fields(decom_data, ccsds_data)

        elif apid in UltraParams.ULTRA_RATES.value.apid:

            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            for packet in sorted_packets:

                decompressed_data = decompress_binary(
                    packet.data["FASTDATA_00"].raw_value,
                    UltraParams.ULTRA_RATES.value.width,
                    UltraParams.ULTRA_RATES.value.block,
                    UltraParams.ULTRA_RATES.value.len_array,
                    UltraParams.ULTRA_RATES.value.mantissa_bit_length,
                )

                for key, item in packet.data.items():
                    # Initialize the list for the first time
                    if key not in decom_data:
                        decom_data[key] = []
                    if key != "FASTDATA_00":
                        decom_data[key].append(item.derived_value)
                    else:
                        decom_data[key].append(decompressed_data)

                ccsds_data = CcsdsData(packet.header)
                parser_helper.append_ccsds_fields(decom_data, ccsds_data)

        else:
            logging.info(f"{apid} is currently not supported")

        return decom_data
