"""
Decommutate a packet file using a given packet definition.

This module contains a common function that can be used by multiple instruments
to decommutate CCSDS packet data using a given XTCE packet definition.
"""

from pathlib import Path
from typing import Union

from space_packet_parser import parser, xtcedef


def decom_packets(
    packet_file: Union[str, Path], xtce_packet_definition: Union[str, Path]
) -> list:
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
    packet_definition = xtcedef.XtcePacketDefinition(xtce_packet_definition)
    packet_parser = parser.PacketParser(packet_definition)

    count = 0

    with open(packet_file, "rb") as binary_data:
        packet_generator = packet_parser.generator(binary_data)

        for packet in packet_generator:
            if packet.header["PKT_APID"].derived_value == 896:
                count += 1
                print(packet.header["SRC_SEQ_CTR"].derived_value)

        return list(packet_generator)
