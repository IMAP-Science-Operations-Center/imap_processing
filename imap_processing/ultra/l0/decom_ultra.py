"""Decommutates Ultra CCSDS packets."""

import logging
from enum import Enum
from typing import NamedTuple

import numpy as np
import xarray as xr

from imap_processing import decom

logging.basicConfig(level=logging.INFO)


class PacketProperties(NamedTuple):
    """Class that represents properties of the ULTRA packet type."""

    apid: int
    width: int
    block: int
    len_array: int


class UltraParams(Enum):
    """Enumerated packet properties for ULTRA."""

    ULTRA_AUX = PacketProperties(apid=880, width=None, block=None, len_array=None)
    ULTRA_IMG_RATES = PacketProperties(apid=881, width=5, block=16, len_array=48)


def read_n_bits(binary: str, n: int, current_position: int):
    """Extract the specified number of bits from a binary string.

    Starting from the current position, it reads n bits. This is used twice.
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
        raise IndexError(
            f"Attempted to read past the end of binary string. "
            f"Current position: {current_position}, "
            f"Requested bits: {n}, String length: {len(binary)}"
        )

    value = int(binary[current_position : current_position + n], 2)
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
    """Decompress a binary string.

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
        if (
            width is None
            or len(decompressed_values) >= UltraParams.ULTRA_IMG_RATES.value.len_array
        ):
            print("hi")
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


def decom_ultra_packets(packet_file: str, xtce: str):
    """
    Unpack and decode ultra packets using CCSDS format and XTCE packet definitions.

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
    packets = decom.decom_packets(packet_file, xtce)

    met_data, science_id, spin_data, abortflag_data, startdelay_data, fastdata_00 = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for packet in packets:
        if (
            packet.header["PKT_APID"].derived_value
            == UltraParams.ULTRA_IMG_RATES.value.apid
        ):
            met_data.append(packet.data["SHCOARSE"].derived_value)
            science_id.append(packet.data["SID"].derived_value)
            spin_data.append(packet.data["SPIN"].derived_value)
            abortflag_data.append(packet.data["ABORTFLAG"].derived_value)
            startdelay_data.append(packet.data["STARTDELAY"].derived_value)
            decompressed_data = decompress_binary(
                packet.data["FASTDATA_00"].raw_value,
                UltraParams.ULTRA_IMG_RATES.value.width,
                UltraParams.ULTRA_IMG_RATES.value.block,
            )
            fastdata_00.append(decompressed_data)

    array_data = np.array(fastdata_00)

    ds = xr.Dataset(
        {
            "science_id": ("epoch", science_id),
            "spin_data": ("epoch", spin_data),
            "abortflag_data": ("epoch", abortflag_data),
            "startdelay_data": ("epoch", startdelay_data),
            "fastdata_00": (("epoch", "index"), array_data),
        },
        coords={
            "epoch": met_data,
        },
    )

    return ds
