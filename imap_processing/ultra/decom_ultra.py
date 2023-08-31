from imap_processing import decom, packet_definition_directory
import logging
from bitstring import ReadError
from space_packet_parser import parser, xtcedef
import copy

logging.basicConfig(level=logging.INFO)


def decom_packets(packet_file: str):
    """Decom Ultra data packets using Ultra packet definition
    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename

    Returns
    -------
    List
        List of all the unpacked data
    """
    xtce_document = f"{packet_definition_directory}ultra/P_U45_IMAGE_RATES.xml"
    return decom_ultra_packets(packet_file, xtce_document)

def safe_packet_generator(packet_parser, binary_data):
    """
    Attempts to generate packets using the provided packet_parser.
    If a ReadError occurs during packet generation, logs the error and stops generation.

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
        for packet in packet_parser.generator(binary_data):
            yield packet
    except ReadError as e:
        logging.error(f"Error reading packet: {e}")

def read_n_bits(binary_blob, n, current_position):
    """Read next n bits from binary_blob starting from current_position."""
    # Ensure we don't read past the end
    if current_position + n > len(binary_blob):
        return None, current_position

    value = int(binary_blob[current_position:current_position + n], 2)
    return value, current_position + n

def log_decompression(value):
    """Perform log decompression on a 16-bit value."""
    e = value >> 12  # Extract the 4 most significant bits for the exponent
    m = value & 0xFFF  # Extract the 12 least significant bits for the mantissa

    if e == 0:
        return m
    else:
        return (4096 + m) << (e - 1)

def decompress_blob(binary_blob):
    """Decompress the given binary blob."""
    current_position = 0
    decompressed_values = []

    while current_position < len(binary_blob):
        # Read the width of the block
        width, current_position = read_n_bits(binary_blob, 5, current_position)

        # If width is None, we don't have enough bits left
        if width is None:
            break

        # If width is 0, we only append a single 0 to the decompressed values
        if width == 0:
            decompressed_values.append(0)
            continue

        # Otherwise, keep reading values of size width until we've read a total of 16*width bits
        # or until we reach the end of the blob
        total_bits_for_values = 16 * width
        end_position = min(current_position + total_bits_for_values, len(binary_blob))

        while current_position < end_position:
            # Make sure we have enough bits left to read the width
            if end_position - current_position < width:
                break
            value, current_position = read_n_bits(binary_blob, width, current_position)

            # If value is None, we don't have enough bits left
            if value is None:
                break

            # Log decompression
            decompressed_value = log_decompression(value)
            decompressed_values.append(decompressed_value)

    return decompressed_values

def decom_ultra_packets(packet_file: str, xtce_packet_definition: str):
    """
    Unpack CCSDS data packet.

    This function unpacks and returns data as it is without any modifications.

    Parameters
    ----------
    packet_file : str
        Path to the data packet file.
    xtce_packet_definition : str
        Path to the XTCE file with its filename.

    Returns
    -------
    list
        A list of all the unpacked data.
    """
    packet_definition = xtcedef.XtcePacketDefinition(xtce_packet_definition)
    packet_parser = parser.PacketParser(packet_definition)

    with open(packet_file, "rb") as binary_data:
        packets = [packet for packet in safe_packet_generator(packet_parser, binary_data)]

    decompressed_packets = []

    for packet in packets:
        # Deep copy the packet to preserve original structure
        new_packet = copy.deepcopy(packet)
        if packet.header['PKT_APID'].derived_value == 881:
            binary_blob = packet.data['FASTDATA_00'].raw_value
            decompressed_data = decompress_blob(binary_blob)
            # Replace only the FASTDATA_00 raw_value with decompressed data
            new_packet.data['FASTDATA_00'].raw_value = decompressed_data
            decompressed_packets.append(new_packet)
        else:
            decompressed_packets.append(packet)

    return decompressed_packets
