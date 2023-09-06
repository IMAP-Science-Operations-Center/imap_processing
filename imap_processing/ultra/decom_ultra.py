from imap_processing import decom
import logging
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
    width_bit = 5  # The bit width that describes the width of data in the block

    while current_position < len(binary):
        # Read the width of the block
        width, current_position = read_n_bits(binary, width_bit, current_position)

        # If width is None, we don't have enough bits left
        if width is None:
            break

        # For each block, read 16 values of the given width
        for _ in range(16):
            # Ensure there are enough bits left to read the width
            if len(binary) - current_position < width:
                break

            value, current_position = read_n_bits(binary, width, current_position)

            # Log decompression and store the value
            decompressed_values.append(log_decompression(value))

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

    packets = decom.decom_packets(packet_file, xtce)

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
            fastdata_00_data.append(decompressed_data)

    fastdata_00_array = np.array(fastdata_00_data, dtype=object)

    ds = xr.Dataset({
        'sid_data': ('time', sid_data),
        'spin_data': ('time', spin_data),
        'abortflag_data': ('time', abortflag_data),
        'startdelay_data': ('time', startdelay_data),
        'fastdata_00': ('time', fastdata_00_array)
    }, coords={
        'time': met_data,
    })

    return ds
