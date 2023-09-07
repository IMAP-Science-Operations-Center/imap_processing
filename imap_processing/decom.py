# Standard
import logging

from bitstring import ReadError

# Installed
from space_packet_parser import parser, xtcedef

logging.basicConfig(level=logging.INFO)


def safe_packet_generator(packet_parser, binary_data):
    """
    Attempts to generate packets using the provided packet_parser.
    If a ReadError occurs during packet generation, logs the error and stops generation.
    This could happen if we are reading off the end of the data (e.g. reading more bits
    than available)

    Parameters
    ----------
    packet_parser : parser.PacketParser
        The parser to use for generating packets.
    binary_data : bytes
        The binary data to parse.

    Yields
    ------
    packet
        The next packet parsed from the binary data.
    """
    try:
        yield from packet_parser.generator(binary_data)
    except ReadError as e:
        logging.error(f"Error reading packet: {e}")

def decom_packets(packet_file: str, xtce_packet_definition: str):
    """Unpack CCSDS data packet. In this function, we unpack and return data
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
        packets = [packet for packet in safe_packet_generator(
            packet_parser, binary_data)]

    return packets
