"""
Decommutate a packet file using a given packet definition.

This module contains a common function that can be used by multiple instruments
to decommutate CCSDS packet data using a given XTCE packet definition.
"""

import logging
from pathlib import Path

import space_packet_parser as spp
from space_packet_parser.exceptions import UnrecognizedPacketTypeError

logger = logging.getLogger(__name__)


def decom_packets(packet_file: str | Path, xtce_packet_definition: str | Path) -> list:
    """
    Unpack CCSDS data packet.

    In this function, we unpack and return data
    as it is. Data modification will not be done at this step.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename.
    xtce_packet_definition : str
        Path to XTCE file with filename.

    Returns
    -------
    list
        List of all the unpacked data.
    """
    packet_definition = spp.load_xtce(xtce_packet_definition)

    with open(packet_file, "rb") as binary_data:
        packets = []
        for binary_packet in spp.ccsds_generator(binary_data):
            try:
                packets.append(packet_definition.parse_bytes(binary_packet))
            except UnrecognizedPacketTypeError as e:
                # Log the error and continue processing other packets
                logger.debug(e)
                continue
        return packets
