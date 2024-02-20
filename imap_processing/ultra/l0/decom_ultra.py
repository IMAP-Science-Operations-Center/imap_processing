"""Decommutates Ultra CCSDS packets."""

import logging
from enum import Enum
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing import decom
from imap_processing.ultra.l0.ultra_utils import EventParser
from imap_processing.ultra.l0.ultra_utils import ULTRAAPID
from imap_processing.utils import group_by_apid, sort_by_time
from imap_processing.ccsds.ccsds_data import CcsdsData
from dataclasses import fields

logging.basicConfig(level=logging.INFO)


class PacketProperties(NamedTuple):
    """Class that represents properties of the ULTRA packet type."""

    apid: int
    width: int
    block: int
    len_array: int
    mantissa_bit_length: int = 12


class UltraParams(Enum):
    """Enumerated packet properties for ULTRA."""

    ULTRA_AUX = PacketProperties(
        apid=880, width=None, block=None, len_array=None, mantissa_bit_length=None)
    ULTRA_IMG_RATES = PacketProperties(
        apid=881, width=5, block=16, len_array=48, mantissa_bit_length=12)
    ULTRA_IMG_ENA_PHXTOF_HI_ANG = PacketProperties(
        apid=883, width=4, block=15, len_array=None, mantissa_bit_length=4)
    ULTRA_IMG_RAW_EVENTS = PacketProperties(
        apid=896, width=None, block=None, len_array=None, mantissa_bit_length=None)


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


def log_decompression(value: int, mantissa_bit_length) -> int:
    """
    Perform logarithmic decompression on an integer, supporting both 16-bit and 8-bit
    formats based on the specified mantissa bit length.

    Parameters
    ----------
    value : int
        An integer comprised of a 4-bit exponent followed by a variable-length mantissa.
    mantissa_bit_length : int
        The bit length of the mantissa (default is 12 for 16-bit format).

    Returns
    -------
    int
        The decompressed integer value.
    """
    # Determine the base value and mask based on mantissa bit length
    if mantissa_bit_length == 12:
        base_value = 4096
        mantissa_mask = 0xFFF
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


def decompress_binary(binary: str, width_bit: int, block: int, len_array, mantissa_bit_length) -> list:
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
    len_array : int
        The length of the array to be decompressed.
    mantissa_bit_length : int
        The bit length of the mantissa.

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
        if width is None or len(decompressed_values) >= len_array:
            break

        # For each block, read 16 values of the given width
        for _ in range(block):
            # Ensure there are enough bits left to read the width
            if len(binary) - current_position < width:
                break

            value, current_position = read_n_bits(binary, width, current_position)

            # Log decompression
            decompressed_values.append(log_decompression(value,
                                                         mantissa_bit_length))

    return decompressed_values


def decompress_image(pp, binary_data, width_bit, mantissa_bit_length,
                  rows=54, cols=180, pixels_per_block=15):
    """
    "Decompresses a binary string representing an image into a matrix of pixel values.
    It starts with an initial pixel value and decompresses the rest of the image using
    block-wise decompression and logarithmic decompression based on provided bit widths and lengths.

    Parameters
    ----------
    pp : int
        The first, unmodified pixel p0,0.
    binary_data : str
        Binary string.
    width_bit : int
        The bit width that describes the width of data in the block
    mantissa_bit_length : int
        The bit length of the mantissa.
    rows : int (Optional)
        Number of rows.
    cols : int (Optional)
        Number of columns.
    pixels_per_block : int (Optional)
        Number of pixels per block.

    Returns
    -------
    p_decom : List of lists.
        Decompressed pixel matrix.

    Notes
    -----
    This process is described starting on page 168 in IMAP-Ultra Flight
    Software Specification document.
    """
    blocks_per_row = int(cols / pixels_per_block)

    # Compressed pixel matrix
    p = [[0 for _ in range(cols)] for _ in range(rows)]
    # Decompressed pixel matrix
    p_decom = [[0 for _ in range(cols)] for _ in range(rows)]

    pos = 0  # Starting position in the binary string

    for i in range(rows):
        for j in range(blocks_per_row):
            # Read the width for the block.
            w, pos = read_n_bits(binary_data, width_bit, pos)
            for k in range(pixels_per_block):
                # Handle the special case in which the width is 0
                if w == 0:
                    value = 0
                else:
                    # Find the value of each pixel in the block
                    value, pos = read_n_bits(binary_data, w, pos)

                # if the least significant bit of value is set (odd)
                if value & 0x01:
                    # value >> 1: shifts bits of value one place to the right
                    # ~: bitwise NOT operator (flips bits)
                    delta_f = ~(value >> 1)
                else:
                    delta_f = value >> 1

                # Calculate the new pixel value and update pp
                column_index = j * pixels_per_block + k
                # 0xff is the hexadecimal representation of the number 255,
                # Keeps only the last 8 bits of the result of pp - delta_f
                # This operation ensures that the result is within the range of an 8-bit byte (0-255)
                p[i][column_index] = (pp - delta_f) & 0xFF
                # Perform logarithmic decompression on the pixel value
                p_decom[i][column_index] = log_decompression(p[i][column_index],
                                                             mantissa_bit_length)
                pp = p[i][column_index]
        pp = p[i][0]

    return p_decom


def read_image_raw_events_binary(packet, events_data=None):
    """
    Converts contents of binary string "EVENTDATA" into values.

    Parameters
    ----------
    packet : space_packet_parser.parser.Packet
        Packet.
    events_data : dict
        Parsed data.

    Returns
    -------
    events_data : dict
        Each for loop appends to the existing dictionary.
    """
    binary = packet.data["EVENTDATA"].raw_value
    count = packet.data["COUNT"].derived_value
    event_length = int(len(binary) / count) if count else 0

    parser = EventParser()

    # Initialize or use the existing events_data structure
    if events_data is None:
        events_data = parser.initialize_event_data(packet.header)

    # Uses fill value for all packets that do not contain event data.
    if count == 0:
        parser.append_fillval(events_data)
        parser.append_values(events_data, packet)
    # For all packets with event data, parses the binary string
    # and appends the other packet values
    else:
        for i in range(count):
            start_index = i * event_length
            event_binary = binary[start_index:start_index + event_length]
            event_data = parser.parse_event(event_binary)

            for key, value in event_data.items():
                events_data[key].append(value)
            parser.append_values(events_data, packet)

    return events_data


def decom_ultra_apids(packet_file: str, xtce: str, test_apid: int = None):
    """
    Unpack and decode ultra packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.
    test_apid : int
        The APID to test. If None, all APIDs are processed.

    Returns
    -------
    xr.Dataset
        A dataset containing the decoded data fields with 'time' as the coordinating
        dimension.
    """
    packets = decom.decom_packets(packet_file, xtce)
    grouped_data = group_by_apid(packets)

    if test_apid:
        grouped_data = {test_apid: grouped_data[test_apid]}

    for apid in grouped_data.keys():

        if (apid == ULTRAAPID.ULTRA_EVENTS_45.value or apid == ULTRAAPID.ULTRA_EVENTS_90.value):
            decom_data = None
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            for packet in sorted_packets:
                decom_data = read_image_raw_events_binary(packet, decom_data)
                count = packet.data["COUNT"].derived_value

                for i in range(count):
                    ccsds_data = CcsdsData(packet.header)
                    for field in fields(CcsdsData):
                        ccsds_key = field.name
                        if ccsds_key not in decom_data:
                            decom_data[ccsds_key] = []
                        decom_data[ccsds_key].append(getattr(ccsds_data, ccsds_key))

        elif (apid == ULTRAAPID.ULTRA_AUX_45.value or apid == ULTRAAPID.ULTRA_AUX_90.value):
            decom_data = {}
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            for packet in sorted_packets:
                for key, item in packet.data.items():
                    if key not in decom_data:
                        decom_data[key] = []
                    decom_data[key].append(item.derived_value)

                ccsds_data = CcsdsData(packet.header)
                for field in fields(CcsdsData):
                    ccsds_key = field.name
                    if ccsds_key not in decom_data:
                        decom_data[ccsds_key] = []
                    decom_data[ccsds_key].append(getattr(ccsds_data, ccsds_key))
        elif (apid == ULTRAAPID.ULTRA_TOF_45.value or apid == ULTRAAPID.ULTRA_TOF_90.value):
            decom_data = {}
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            for packet in sorted_packets:
                decompressed_data = decompress_image(
                    packet.data["P00"].derived_value,
                    packet.data["PACKETDATA"].raw_value,
                    UltraParams.ULTRA_IMG_ENA_PHXTOF_HI_ANG
                    .value.width,
                    UltraParams.ULTRA_IMG_ENA_PHXTOF_HI_ANG
                    .value.mantissa_bit_length)

                for key, item in packet.data.items():
                    if key not in decom_data:
                        decom_data[key] = []
                    if key != "PACKETDATA":
                        decom_data[key].append(item.derived_value)
                    else:
                        decom_data[key].append(decompressed_data)

                ccsds_data = CcsdsData(packet.header)
                for field in fields(CcsdsData):
                    ccsds_key = field.name
                    if ccsds_key not in decom_data:
                        decom_data[ccsds_key] = []
                    decom_data[ccsds_key].append(getattr(ccsds_data, ccsds_key))

        elif (apid == ULTRAAPID.ULTRA_RATES_45.value or
              apid == ULTRAAPID.ULTRA_RATES_90.value):
            decom_data = {}
            sorted_packets = sort_by_time(grouped_data[apid], "SHCOARSE")

            for packet in sorted_packets:

                decompressed_data = decompress_binary(
                    packet.data["FASTDATA_00"].raw_value,
                    UltraParams.ULTRA_IMG_RATES.value.width,
                    UltraParams.ULTRA_IMG_RATES.value.block,
                    UltraParams.ULTRA_IMG_RATES.value.len_array,
                    UltraParams.ULTRA_IMG_RATES.value.mantissa_bit_length,
                )

                for key, item in packet.data.items():
                    if key not in decom_data:
                        decom_data[key] = []
                    if key != "FASTDATA_00":
                        decom_data[key].append(item.derived_value)
                    else:
                        decom_data[key].append(decompressed_data)

                ccsds_data = CcsdsData(packet.header)
                for field in fields(CcsdsData):
                    ccsds_key = field.name
                    if ccsds_key not in decom_data:
                        decom_data[ccsds_key] = []
                    decom_data[ccsds_key].append(getattr(ccsds_data, ccsds_key))

        else:
            logging.info(f"{apid} is currently not supported")

        if (apid == ULTRAAPID.ULTRA_TOF_45.value or apid == ULTRAAPID.ULTRA_TOF_90.value):
            array_data = np.array(decom_data['PACKETDATA'])

            multi_index = pd.MultiIndex.from_arrays(
                [decom_data['SHCOARSE'], decom_data['SID']], names=("epoch", "science_id")
            )

            ds = xr.Dataset(
                {
                    "spin_data": ("measurement", decom_data['SPIN']),
                    "abortflag_data": ("measurement", decom_data['ABORTFLAG']),
                    "startdelay_data": ("measurement", decom_data['STARTDELAY']),
                    "p00_data": ("measurement", decom_data['P00']),
                    "packetdata": (("measurement", "row", "col"), array_data),
                },
            coords={
                "measurement": multi_index,
            },
        )

        elif (apid == ULTRAAPID.ULTRA_RATES_45.value
              or apid == ULTRAAPID.ULTRA_RATES_90.value):

            array_data = np.array(decom_data['FASTDATA_00'])

            ds = xr.Dataset(
                {
                    "science_id": ("epoch", decom_data['SID']),
                    "spin_data": ("epoch", decom_data['SPIN']),
                    "abortflag_data": ("epoch", decom_data['ABORTFLAG']),
                    "startdelay_data": ("epoch", decom_data['STARTDELAY']),
                    "fastdata_00": (("epoch", "index"), array_data),
                },
                coords={
                    "epoch": decom_data['SHCOARSE'],
                },
            )


        elif (apid == ULTRAAPID.ULTRA_EVENTS_45.value or apid == ULTRAAPID.ULTRA_EVENTS_90.value
              or apid == ULTRAAPID.ULTRA_AUX_45.value or apid == ULTRAAPID.ULTRA_AUX_90.value):

            data_arrays = {}

            for key, values in decom_data.items():
                if key != "SHCOARSE":
                    data_arrays[key] = xr.DataArray(values, dims=["epoch"])

            ds = xr.Dataset(
                data_vars=data_arrays,
                coords={"epoch": decom_data['SHCOARSE']},
            )

    return ds
