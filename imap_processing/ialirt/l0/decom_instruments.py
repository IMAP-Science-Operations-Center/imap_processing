import logging
from enum import Enum
from typing import NamedTuple

import numpy as np
import xarray as xr

from imap_processing.ialirt.l0 import decom_ialirt

logging.basicConfig(level=logging.INFO)


class PacketProperties(NamedTuple):
    """Class that represents properties of the ULTRA packet type."""
    apid: int
    width: int
    block: int
    len_array: int


class UltraParams(Enum):
    ULTRA_AUX = PacketProperties(apid=880, width=None, block=None,
                                 len_array=None)
    ULTRA_IMG_RATES = PacketProperties(apid=881, width=5, block=16,
                                       len_array = 48)


def read_n_bits(binary: str, n: int, current_position: int):
    """
    Extract a specified number of bits from a binary string,
    starting from a given position. This is used twice.
    The first time it reads the first 5 bits to determine the width.
    The second time it uses the width to determine the value of the bitstring.

    Parameters
    ----------
    binary : str
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
    if current_position + n > len(binary):
        raise IndexError(f"Attempted to read past the end of binary string. "
                         f"Current position: {current_position}, "
                         f"Requested bits: {n}, String length: {len(binary)}")

    value = int(binary[current_position:current_position + n], 2)
    return value, current_position + n


def log_decompression(value: int) -> int:
    """
    Perform logarithmic decompression on a 16-bit integer.

    Parameters
    ----------
    value : int
        A 16-bit integer comprised of a 4-bit exponent followed by a 12-bit mantissa.

    Returns
    -------
    int
        The decompressed integer value.

    Note: Equations from Section 1.2.1.1 Data Compression and Decompression Algorithms
    in Ultra_algorithm_doc_rev2.pdf.
    """
    # The exponent e, and mantissa, m are 4-bit and 12-bit unsigned integers
    # respectively
    e = value >> 12  # Extract the 4 most significant bits for the exponent
    m = value & 0xFFF  # Extract the 12 least significant bits for the mantissa

    if e == 0:
        return m
    else:
        return (4096 + m) << (e - 1)


def decompress_binary(binary: str, width_bit: int, block: int) -> list:
    """
    Decompress a binary string based on block-width encoding and
    logarithmic compression.

    This function interprets a binary string where the first 'width_bit' bits
    specifies the width of the following values. Each value is then extracted and
    subjected to logarithmic decompression.

    Parameters
    ----------
    binary : str
        A binary string containing the compressed data.
    width_bit : int
        The bit width that describes the width of data in the block
    block : int
        Number of values in each block

    Returns
    -------
    list
        A list of decompressed values.

    Note: Equations from Section 1.2.1.1 Data Compression and Decompression Algorithms
    in Ultra_algorithm_doc_rev2.pdf.
    """
    current_position = 0
    decompressed_values = []

    while current_position < len(binary):
        # Read the width of the block
        width, current_position = read_n_bits(binary, width_bit, current_position)
        # If width is 0 or None, we don't have enough bits left
        if width is None or len(decompressed_values) >= \
                UltraParams.ULTRA_IMG_RATES.value.len_array:
            print('hi')
            break

        # For each block, read 16 values of the given width
        for _ in range(block):
            # Ensure there are enough bits left to read the width
            if len(binary) - current_position < width:
                break

            value, current_position = read_n_bits(binary, width, current_position)

            # Log decompression
            decompressed_values.append(log_decompression(value))

    return decompressed_values


def decom_instrument_packets(packet_file: str, xtce: str):
    """
    Unpack and decode hit packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.

    Returns
    -------
    xr.Dataset
        A dataset containing the decoded data fields with 'time' as the coordinating
        dimension.
    """

    packets = decom_ialirt.decom_packets(packet_file, xtce)

    hit_storage = {}

    for packet in packets:
        for key, value in packet.data.items():
            if key.startswith('HIT'):
                if key not in hit_storage:
                    hit_storage[key] = []
                hit_storage[key].append(value.derived_value)

    hit_ds = xr.Dataset({
        'status': ('met', hit_storage['HIT_STATUS']),
        'reserved': ('met', hit_storage['HIT_RESERVED']),
        'counter': ('met', hit_storage['HIT_COUNTER']),
        'fast_rate_1': ('met', hit_storage['HIT_FAST_RATE_1']),
        'fast_rate_2': ('met', hit_storage['HIT_FAST_RATE_2']),
        'slow_rate': ('met', hit_storage['HIT_SLOW_RATE']),
        'event_data_01': ('met', hit_storage['HIT_EVENT_DATA_01']),
        'event_data_02': ('met', hit_storage['HIT_EVENT_DATA_02']),
        'event_data_03': ('met', hit_storage['HIT_EVENT_DATA_03']),
        'event_data_04': ('met', hit_storage['HIT_EVENT_DATA_04']),
        'event_data_05': ('met', hit_storage['HIT_EVENT_DATA_05']),
        'event_data_06': ('met', hit_storage['HIT_EVENT_DATA_06']),
        'event_data_07': ('met', hit_storage['HIT_EVENT_DATA_07']),
        'event_data_08': ('met', hit_storage['HIT_EVENT_DATA_08']),
        'event_data_09': ('met', hit_storage['HIT_EVENT_DATA_09']),
        'event_data_10': ('met', hit_storage['HIT_EVENT_DATA_10']),
    }, coords={
        'met': hit_storage['HIT_MET'],
    })

    return hit_ds
