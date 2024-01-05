"""Decommutate a packet file using a given packet definition.

This module contains a common function that can be used by multiple instruments
to decommutate CCSDS packet data using a given XTCE packet definition.
"""

from space_packet_parser import parser, xtcedef


def decom_packets(packet_file: str, xtce_packet_definition: str):
    """Unpack CCSDS data packet.

    In this function, we unpack and return data
    as it is. Data modification will not be done at this step.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename
    xtce_packet_definition : str
        Path to XTCE file with filename

    Returns
    -------
    List
        List of all the unpacked data
    """
    packet_definition = xtcedef.XtcePacketDefinition(xtce_packet_definition)
    packet_parser = parser.PacketParser(packet_definition)

    with open(packet_file, "rb") as binary_data:
        packet_generator = packet_parser.generator(binary_data)
        return list(packet_generator)
