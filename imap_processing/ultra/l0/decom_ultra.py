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
    ULTRA_IMG_ENA_PHXTOF_HI_ANG = PacketProperties(
        apid=883, width=4, block=15, len_array=None
    )
    ULTRA_IMG_RAW_EVENTS = PacketProperties(
        apid=896, width=None, block=None, len_array=None
    )


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


def log_decompression_8bit(value: int) -> int:
    """
    Perform logarithmic decompression on an 8-bit integer based on a 4-bit exponent
    and a 4-bit mantissa. The exponent is always less than 13.

    Parameters
    ----------
    value : int
        An 8-bit integer comprised of a 4-bit exponent followed by a 4-bit mantissa.

    Returns
    -------
    int
        The decompressed integer value.
    """
    # The exponent e is 4 bits, and the mantissa m is 4 bits
    e = value >> 4  # Extract the 4 most significant bits for the exponent
    m = value & 0x0F  # Extract the 4 least significant bits for the mantissa

    if e == 0:
        return m
    else:
        return (16 + m) << (e - 1)


def decompress_binary(binary: str, width_bit: int, block: int, len_array) -> list:
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
        if width is None or len(decompressed_values) >= len_array:
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


def decom_ultra_img_rates_packets(packet_file: str, xtce: str):
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
        [] for _ in range(6)
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
                UltraParams.ULTRA_IMG_RATES.value.len_array,
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


def process_image(pp, binary_data, rows, cols, blocks_per_row, pixels_per_block):
    # p[53][179]
    p = [[0 for _ in range(cols)] for _ in range(rows)]
    final = [
        [0 for _ in range(cols)] for _ in range(rows)
    ]  # Initialize the pixel matrix
    pos = 0  # Starting position in the binary string

    for i in range(rows):
        for j in range(blocks_per_row):
            w, pos = read_n_bits(binary_data, 4, pos)  # Read the width for the block
            for k in range(pixels_per_block):
                if w == 0:  # Handle the special case where read(0) should return 0
                    value = 0
                else:
                    value, pos = read_n_bits(
                        binary_data, w, pos
                    )  # Read the Δcode using the width w

                if value & 0x01:  # if the least significant bit of value is set (odd)
                    # value >> 1: shifts bits of value one place to the right
                    # ~: bitwise NOT operator (flips bits)
                    delta_f = ~(value >> 1)
                else:
                    delta_f = value >> 1

                # Calculate the new pixel value and update pp
                column_index = j * pixels_per_block + k
                # 0xff is the hexadecimal representation of the number 255,
                # keeps only the last 8 bits of the result of pp - delta_f and discards all other higher bits
                # This operation ensures that the result is within the range of an 8-bit byte (0-255)
                p[i][column_index] = (pp - delta_f) & 0xFF
                final[i][column_index] = log_decompression_8bit(p[i][column_index])
                pp = p[i][column_index]
        pp = p[i][0]

    return final


def decom_image_ena_phxtof_hi_ang_packets(packet_file: str, xtce: str):
    packets = decom.decom_packets(packet_file, xtce)

    (
        met_data,
        science_id,
        spin_data,
        abortflag_data,
        startdelay_data,
        p00_data,
        packetdata,
    ) = ([] for _ in range(7))

    for packet in packets:
        if (
            packet.header["PKT_APID"].derived_value
            == UltraParams.ULTRA_IMG_ENA_PHXTOF_HI_ANG.value.apid
        ):
            met_data.append(packet.data["SHCOARSE"].derived_value)
            science_id.append(packet.data["SID"].derived_value)
            spin_data.append(packet.data["SPIN"].derived_value)
            abortflag_data.append(packet.data["ABORTFLAG"].derived_value)
            startdelay_data.append(packet.data["STARTDELAY"].derived_value)
            p00_data.append(packet.data["P00"].derived_value)
            decompressed_data = process_image(
                packet.data["P00"].derived_value,
                packet.data["PACKETDATA"].raw_value,
                54,
                180,
                int(180 / 15),
                15,
            )
            packetdata.append(decompressed_data)

    array_data = np.array(packetdata)

    multi_index = pd.MultiIndex.from_arrays(
        [met_data, science_id], names=("epoch", "science_id")
    )

    ds = xr.Dataset(
        {
            "spin_data": ("measurement", spin_data),
            "abortflag_data": ("measurement", abortflag_data),
            "startdelay_data": ("measurement", startdelay_data),
            "p00_data": ("measurement", p00_data),
            "packetdata": (("measurement", "row", "col"), array_data),
        },
        coords={
            "measurement": multi_index,
        },
    )

    return ds


def decom_ultra_aux_packets(packet_file: str, xtce: str):
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

    (
        met_data,
        spin_period_valid,
        spin_phase_valid,
        spin_period_source,
        catbed_heater_flag,
        hw_mode,
        imc_enb,
        left_deflection_charge,
        right_deflection_charge,
        time_spin_start,
        time_spin_start_sub,
        duration,
        spin_number,
        time_spin_data,
        spin_period,
        spin_phase,
    ) = ([] for _ in range(16))

    for packet in packets:
        met_data.append(packet.data["SHCOARSE"].derived_value)
        spin_period_valid.append(packet.data["SPINPERIODVALID"].derived_value)
        spin_phase_valid.append(packet.data["SPINPHASEVALID"].derived_value)
        spin_period_source.append(packet.data["SPINPERIODSOURCE"].derived_value)
        catbed_heater_flag.append(packet.data["CATBEDHEATERFLAG"].derived_value)
        hw_mode.append(packet.data["HWMODE"].derived_value)
        imc_enb.append(packet.data["IMCENB"].derived_value)
        left_deflection_charge.append(packet.data["LEFTDEFLECTIONCHARGE"].derived_value)
        right_deflection_charge.append(
            packet.data["RIGHTDEFLECTIONCHARGE"].derived_value
        )
        time_spin_start.append(packet.data["TIMESPINSTART"].derived_value)
        time_spin_start_sub.append(packet.data["TIMESPINSTARTSUB"].derived_value)
        duration.append(packet.data["DURATION"].derived_value)
        spin_number.append(packet.data["SPINNUMBER"].derived_value)
        time_spin_data.append(packet.data["TIMESPINDATA"].derived_value)
        spin_period.append(packet.data["SPINPERIOD"].derived_value)
        spin_phase.append(packet.data["SPINPHASE"].derived_value)

    ds = xr.Dataset(
        {
            "spin_period_valid": ("epoch", spin_period_valid),
            "spin_phase_valid": ("epoch", spin_phase_valid),
            "spin_period_source": ("epoch", spin_period_source),
            "catbed_heater_flag": ("epoch", catbed_heater_flag),
            "hw_mode": ("epoch", hw_mode),
            "imc_enb": ("epoch", imc_enb),
            "left_deflection_charge": ("epoch", left_deflection_charge),
            "right_deflection_charge": ("epoch", right_deflection_charge),
            "time_spin_start": ("epoch", time_spin_start),
            "time_spin_start_sub": ("epoch", time_spin_start_sub),
            "duration": ("epoch", duration),
            "spin_number": ("epoch", spin_number),
            "time_spin_data": ("epoch", time_spin_data),
            "spin_period": ("epoch", spin_period),
            "spin_phase": ("epoch", spin_phase),
        },
        coords={
            "epoch": met_data,
        },
    )

    return ds


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

    # Uses -1 for all packets that do not contain event data.
    if count == 0:
        parser.append_negative_one(events_data)
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


def decom_image_raw_events_packets(packet_file: str, xtce: str):
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
    events_data = None  # Initialize to None for the first packet

    for packet in packets:
        if (
                packet.header["PKT_APID"].derived_value == ULTRAAPID.ULTRA_EVENTS_45.value or
                packet.header["PKT_APID"].derived_value == ULTRAAPID.ULTRA_EVENTS_90.value
        ):
            events_data = read_image_raw_events_binary(packet, events_data)

    data_arrays = {}

    for key, values in events_data.items():
        if key != "SHCOARSE":
            data_arrays[key] = xr.DataArray(values, dims=["epoch"])

    ds = xr.Dataset(
        data_vars=data_arrays,
        coords={"epoch": events_data['SHCOARSE']},
    )

    return ds
