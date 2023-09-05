from imap_processing import decom, packet_definition_directory
import logging
from bitstring import ReadError
from space_packet_parser import parser, xtcedef
import copy
import xarray as xr
import numpy as np

logging.basicConfig(level=logging.INFO)

def read_n_bits(binary, n, current_position):
    """Read next n bits from binary string starting from current_position."""
    # Ensure we don't read past the end
    if current_position + n > len(binary):
        return None, current_position

    value = int(binary[current_position:current_position + n], 2)
    return value, current_position + n

def log_decompression(value):
    """Perform log decompression on a 16-bit value."""
    # The exponent e, and mantissa, m are 4-bit and 12-bit unsigned integers respectively
    e = value >> 12  # Extract the 4 most significant bits for the exponent
    m = value & 0xFFF  # Extract the 12 least significant bits for the mantissa

    if e == 0:
        return m
    else:
        return (4096 + m) << (e - 1)

def decompress_binary(binary):
    """Decompress the given binary."""
    current_position = 0
    decompressed_values = []

    while current_position < len(binary):
        # Read the width of the block
        width, current_position = read_n_bits(binary, 5, current_position)

        # If width is None, we don't have enough bits left
        if width is None:
            break

        # Otherwise, keep reading values of size width until we've read a total of 16*width bits
        # or until we reach the end of the binary.
        total_bits_for_values = 16 * width
        end_position = min(current_position + total_bits_for_values, len(binary))

        while current_position < end_position:
            # Make sure we have enough bits left to read the width
            if end_position - current_position < width:
                break
            value, current_position = read_n_bits(binary, width, current_position)

            # Log decompression
            decompressed_value = log_decompression(value)
            decompressed_values.append(decompressed_value)

    return decompressed_values

def decom_ultra_packets(packet_file: str, xtce: str):
    """
    Unpack CCSDS data packet.

    This function unpacks and returns data.

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

    xtce_document = f"{packet_definition_directory}ultra/{xtce}"
    packets = decom.decom_packets(packet_file, xtce_document)

    met_data, sid_data, spin_data, abortflag_data, startdelay_data, fastdata_00_data = \
        [], [], [], [], [], []

    for packet in packets:
        if packet.header['PKT_APID'].derived_value == 881:
            met_data.append(packet.data['SHCOARSE'].derived_value)
            sid_data.append(packet.data['SID'].derived_value)
            spin_data.append(packet.data['SPIN'].derived_value)
            abortflag_data.append(packet.data['ABORTFLAG'].derived_value)
            startdelay_data.append(packet.data['STARTDELAY'].derived_value)
            decompressed_data = decompress_binary(packet.data['FASTDATA_00'].raw_value)
            fastdata_00_data.append(decompressed_data[:48])

    # Create Dataset directly with all data variables
    ds = xr.Dataset({
        'sid_data': ('time', sid_data),
        'spin_data': ('time', spin_data),
        'abortflag_data': ('time', abortflag_data),
        'startdelay_data': ('time', startdelay_data),
        'fastdata_00': (['time', 'measurement'], fastdata_00_data)
    }, coords={
        'time': met_data,
        'measurement': np.arange(48)
    })

    return ds
