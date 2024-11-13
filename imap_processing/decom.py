"""
Decommutate a packet file using a given packet definition.

This module contains a common function that can be used by multiple instruments
to decommutate CCSDS packet data using a given XTCE packet definition.
"""

from pathlib import Path
from typing import Union

from space_packet_parser import definitions


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
    packet_definition = definitions.XtcePacketDefinition(xtce_packet_definition)

    with open(packet_file, "rb") as binary_data:
        packet_generator = packet_definition.packet_generator(binary_data)
        return list(packet_generator)
