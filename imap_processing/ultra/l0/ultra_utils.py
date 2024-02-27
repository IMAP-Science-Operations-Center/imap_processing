"""Contains data classes to support Ultra L0 processing."""
from dataclasses import fields
from typing import NamedTuple

from imap_processing.cdf.defaults import GlobalConstants


class PacketProperties(NamedTuple):
    """Class that represents properties of the ULTRA packet type."""

    apid: list  # List of APIDs
    width: int  # Width of binary data
    block: int  # Number of values in each block.
    # This is important for decompressing the images and
    # a description is available on page 171 of IMAP-Ultra Flight
    # Software Specification document (7523-9009_Rev_-.pdf).
    len_array: int  # Length of the array to be decompressed
    mantissa_bit_length: int  # used to determine the level of
    # precision that can be recovered from compressed data.


# Define PacketProperties instances directly in the module namespace
ULTRA_AUX = PacketProperties(
    apid=[880, 994],
    width=None,
    block=None,
    len_array=None,
    mantissa_bit_length=None,
)
ULTRA_RATES = PacketProperties(
    apid=[881, 945], width=5, block=16, len_array=48, mantissa_bit_length=12
)
ULTRA_TOF = PacketProperties(
    apid=[883, 947], width=4, block=15, len_array=None, mantissa_bit_length=4
)
ULTRA_EVENTS = PacketProperties(
    apid=[896, 960],
    width=None,
    block=None,
    len_array=None,
    mantissa_bit_length=None,
)


# Module-level constant for event field ranges
EVENT_FIELD_RANGES = {
    # Coincidence Type
    "coin_type": (0, 2),
    # Start Type
    "start_type": (2, 4),
    # Stop Type
    "stop_type": (4, 8),
    # Start Position Time to Digital Converter
    "start_pos_tdc": (8, 19),
    # Stop North Time to Digital Converter
    "stop_north_tdc": (19, 30),
    # Stop East Time to Digital Converter
    "stop_east_tdc": (30, 41),
    # Stop South Time to Digital Converter
    "stop_south_tdc": (41, 52),
    # Stop West Time to Digital Converter
    "stop_west_tdc": (52, 63),
    # Coincidence North Time to Digital Converter
    "coin_north_tdc": (63, 74),
    # Coincidence South Time to Digital Converter
    "coin_south_tdc": (74, 85),
    # Coincidence Discrete Time to Digital Converter
    "coin_discrete_tdc": (85, 96),
    # Energy/Pulse Height
    "energy_ph": (96, 108),
    # Pulse Width
    "pulse_width": (108, 119),
    # Event Flag Count
    "event_flag_cnt": (119, 120),
    # Event Flag PHCmpSL
    "event_flag_phcmpsl": (120, 121),
    # Event Flag PHCmpSR
    "event_flag_phcmpsr": (121, 122),
    # Event Flag PHCmpCD
    "event_flag_phcmpcd": (122, 123),
    # Solid State Detector Flags
    "ssd_flag_7": (123, 124),
    "ssd_flag_6": (124, 125),
    "ssd_flag_5": (125, 126),
    "ssd_flag_4": (126, 127),
    "ssd_flag_3": (127, 128),
    "ssd_flag_2": (128, 129),
    "ssd_flag_1": (129, 130),
    "ssd_flag_0": (130, 131),
    # Constant Fraction Discriminator Flag Coincidence Top North
    "cfd_flag_cointn": (
        131,
        132,
    ),
    # Constant Fraction Discriminator Flag Coincidence Bottom North
    "cfd_flag_coinbn": (
        132,
        133,
    ),
    # Constant Fraction Discriminator Flag Coincidence Top South
    "cfd_flag_coints": (
        133,
        134,
    ),
    # Constant Fraction Discriminator Flag Coincidence Bottom South
    "cfd_flag_coinbs": (
        134,
        135,
    ),
    # Constant Fraction DiscriminatorFlag Coincidence Discrete
    "cfd_flag_coind": (
        135,
        136,
    ),
    # Constant Fraction Discriminator Flag Start Right Full
    "cfd_flag_startrf": (
        136,
        137,
    ),
    # Constant Fraction Discriminator Flag Start Left Full
    "cfd_flag_startlf": (
        137,
        138,
    ),
    # Constant Fraction Discriminator Flag Start Position Right
    "cfd_flag_startrp": (
        138,
        139,
    ),
    # Constant Fraction Discriminator Flag Start Position Left
    "cfd_flag_startlp": (
        139,
        140,
    ),
    # Constant Fraction Discriminator Flag Stop Top North
    "cfd_flag_stoptn": (
        140,
        141,
    ),
    # Constant Fraction Discriminator Flag Stop Bottom North
    "cfd_flag_stopbn": (
        141,
        142,
    ),
    # Constant Fraction Discriminator Flag Stop Top East
    "cfd_flag_stopte": (142, 143),
    # Constant Fraction Discriminator Flag Stop Bottom East
    "cfd_flag_stopbe": (
        143,
        144,
    ),
    # Constant Fraction Discriminator Flag Stop Top South
    "cfd_flag_stopts": (
        144,
        145,
    ),
    # Constant Fraction Discriminator Flag Stop Bottom South
    "cfd_flag_stopbs": (
        145,
        146,
    ),
    # Constant Fraction Discriminator Flag Stop Top West
    "cfd_flag_stoptw": (146, 147),
    # Constant Fraction Discriminator Flag Stop Bottom West
    "cfd_flag_stopbw": (
        147,
        148,
    ),
    "bin": (148, 156),  # Bin
    "phase_angle": (156, 166),  # Phase Angle
}


def append_fillval(decom_data: dict, packet):
    """Append fill values to all fields.

    Parameters
    ----------
    decom_data : dict
        Parsed data.
    packet : space_packet_parser.parser.Packet
        Packet.
    """
    for key in decom_data:
        if (key not in packet.header.keys()) and (key not in packet.data.keys()):
            decom_data[key].append(GlobalConstants.INT_FILLVAL)


def parse_event(event_binary):
    """Parse a binary string representing a single event.

    Parameters
    ----------
    event_binary : str
        Event binary string.
    """
    fields_dict = {}
    for field, (start, end) in EVENT_FIELD_RANGES.items():
        field_value = int(event_binary[start:end], 2)
        fields_dict[field] = field_value
    return fields_dict


def append_ccsds_fields(decom_data: dict, ccsds_data_object: object):
    """Append CCSDS fields to event_data.

    Parameters
    ----------
    decom_data : dict
        Parsed data.
    ccsds_data_object : object
        CCSDS data object.
    """
    for field in fields(ccsds_data_object.__class__):
        ccsds_key = field.name
        if ccsds_key not in decom_data:
            decom_data[ccsds_key] = []
        decom_data[ccsds_key].append(getattr(ccsds_data_object, ccsds_key))
