"""Ultra Decompression Tools."""

import numpy as np
from numpy.typing import NDArray

from imap_processing.ultra.l0.ultra_utils import (
    parse_event,
)
from imap_processing.utils import convert_to_binary_string


def read_and_advance(
    binary_data: str, n: int, current_position: int
) -> tuple[int, int]:
    """
    Extract the specified number of bits from a binary string.

    Starting from the current position, it reads n bits. This is used twice.
    The first time it reads the first 5 bits to determine the width.
    The second time it uses the width to determine the value of the bitstring.

    Parameters
    ----------
    binary_data : str
        The string of binary data from which bits will be read.
        This is a string of 0's and 1's.
    n : int
        Number of bits to read from the binary string.
    current_position : int
        The starting position in the binary string from which bits will be read.

    Returns
    -------
    value : int
        The integer representation of the read bits or None if the end of the
        string is reached before reading 'n' bits.
    current_position + n
        - The updated position in the binary string after reading the bits.
    """
    # Ensure we don't read past the end
    if current_position + n > len(binary_data):
        raise IndexError(
            f"Attempted to read past the end of binary string. "
            f"Current position: {current_position}, "
            f"Requested bits: {n}, String length: {len(binary_data)}"
        )

    value = int(binary_data[current_position : current_position + n], 2)
    return value, current_position + n


def log_decompression(value: int, mantissa_bit_length: int) -> int:
    """
    Perform logarithmic decompression on an integer.

    Supports 16-bit, 10-bit, and 8-bit formats based on the specified
    mantissa bit length.

    Parameters
    ----------
    value : int
        An integer comprised of an exponent followed by a mantissa.
    mantissa_bit_length : int
        The bit length of the mantissa.

    Returns
    -------
    int
        The decompressed integer value.
    """
    # Determine the base value and mask based on mantissa bit length
    if mantissa_bit_length == 12:
        base_value = 4096
        mantissa_mask = 0xFFF
    elif mantissa_bit_length == 5:
        base_value = 32
        mantissa_mask = 0x1F
    elif mantissa_bit_length == 4:
        base_value = 16
        mantissa_mask = 0x0F
    else:
        raise ValueError("Unsupported mantissa bit length")

    # Extract the exponent and mantissa
    e = value >> mantissa_bit_length  # Extract the exponent
    m = value & mantissa_mask  # Extract the mantissa

    if e == 0:
        return m
    else:
        return (base_value + m) << (e - 1)


def decompress_binary(
    binary: str, width_bit: int, block: int, array_length: int, mantissa_bit_length: int
) -> list:
    """
    Will decompress a binary string.

    Decompress a binary string based on block-width encoding and
    logarithmic compression.

    This function interprets a binary string where the value of 'width_bits'
    specifies the width of the following bits. Each value is then extracted and
    subjected to logarithmic decompression.

    Parameters
    ----------
    binary : str
        A binary string containing the compressed data.
    width_bit : int
        The bit width that describes the width of data in the block.
    block : int
        Number of values in each block.
    array_length : int
        The length of the array to be decompressed.
    mantissa_bit_length : int
        The bit length of the mantissa.

    Returns
    -------
    list
        A list of decompressed values.

    Notes
    -----
    Equations from Section 1.2.1.1 Data Compression and Decompression Algorithms
    in Ultra_algorithm_doc_rev2.pdf.
    """
    current_position = 0
    decompressed_values: list = []

    while current_position < len(binary) and len(decompressed_values) < array_length:
        # Read the width of the block
        width, current_position = read_and_advance(binary, width_bit, current_position)
        # If width is 0, add 'block' number of zeroes and continue
        if width == 0:
            decompressed_values.extend([0] * block)
            continue

        # For each block, read 'block' values of the given width
        for _ in range(block):
            # Ensure there are enough bits left to read the width
            if len(binary) - current_position < width:
                break

            value, current_position = read_and_advance(binary, width, current_position)

            # Log decompression
            decompressed_values.append(log_decompression(value, mantissa_bit_length))

    return decompressed_values


def decompress_image(
    pixel0: int,
    binary_data: str,
    width_bit: int,
    mantissa_bit_length: int,
) -> NDArray:
    """
    Will decompress a binary string representing an image into a matrix of pixel values.

    It starts with an initial pixel value and decompresses the rest of the image using
    block-wise decompression and logarithmic decompression based on provided bit widths
    and lengths.

    Parameters
    ----------
    pixel0 : int
        The first, unmodified pixel p0,0.
    binary_data : str
        Binary string.
    width_bit : int
        The bit width that describes the width of data in the block.
    mantissa_bit_length : int
        The bit length of the mantissa.

    Returns
    -------
    p_decom : NDArray
        A 2D numpy array representing pixel values.
        Each pixel is stored as an unsigned 16-bit integer (uint16).

    Notes
    -----
    This process is described starting on page 168 in IMAP-Ultra Flight
    Software Specification document.
    """
    rows = 54
    cols = 180
    pixels_per_block = 15

    blocks_per_row = cols // pixels_per_block

    # Compressed pixel matrix
    p = np.zeros((rows, cols), dtype=np.uint16)
    # Decompressed pixel matrix
    p_decom = np.zeros((rows, cols), dtype=np.int16)

    pos = 0  # Starting position in the binary string

    for i in range(rows):
        for j in range(blocks_per_row):
            # Read the width for the block.
            w, pos = read_and_advance(binary_data, width_bit, pos)
            for k in range(pixels_per_block):
                # Handle the special case in which the width is 0
                if w == 0:
                    value = 0
                else:
                    # Find the value of each pixel in the block
                    value, pos = read_and_advance(binary_data, w, pos)

                # if the least significant bit of value is set (odd)
                if value & 0x01:
                    # value >> 1: shifts bits of value one place to the right
                    # ~: bitwise NOT operator (flips bits)
                    delta_f = ~(value >> 1)
                else:
                    delta_f = value >> 1

                # Calculate the new pixel value and update pixel0
                column_index = j * pixels_per_block + k
                # 0xff is the hexadecimal representation of the number 255,
                # Keeps only the last 8 bits of the result of pixel0 - delta_f
                # This operation ensures that the result is within the range
                # of an 8-bit byte (0-255)
                # Use np.int16 for the arithmetic operation to avoid overflow
                # Then implicitly cast back to the p's uint16 dtype for storage
                p[i][column_index] = np.int16(pixel0) - delta_f
                # Perform logarithmic decompression on the pixel value
                p_decom[i][column_index] = log_decompression(
                    p[i][column_index], mantissa_bit_length
                )
                pixel0 = p[i][column_index]
        pixel0 = p[i][0]

    return p_decom


def read_image_raw_events_binary(
    event_data: bytes,
    count: int,
    field_ranges: dict,
) -> NDArray:
    """
    Convert contents of binary string 'EVENTDATA' into values.

    Parameters
    ----------
    event_data : bytes
        Event data.
    count : int
        Number of events.
    field_ranges : dict
        Field ranges for the event data.

    Returns
    -------
    event_data : NDArray
        Event data.
    """
    binary = convert_to_binary_string(event_data)
    length = max(end for (_, end) in field_ranges.values())
    # bits per event
    event_length = length if count else 0
    event_data_list = []

    # For all packets with event data, parses the binary string
    for i in range(count):
        start_index = i * event_length
        event_binary = binary[start_index : start_index + event_length]
        parsed_event = parse_event(event_binary, field_ranges)
        event_data_list.append(parsed_event)

    return np.array(event_data_list)
