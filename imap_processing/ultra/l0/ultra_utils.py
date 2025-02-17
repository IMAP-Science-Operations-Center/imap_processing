"""Contains data classes to support Ultra L0 processing."""

from dataclasses import fields
from typing import NamedTuple, Union

import numpy as np


class PacketProperties(NamedTuple):
    """Class that represents properties of the ULTRA packet type."""

    apid: list  # List of APIDs
    logical_source: list  # List of logical sources
    addition_to_logical_desc: str  # Description of the logical source
    width: Union[int, None]  # Width of binary data (could be None).
    block: Union[int, None]  # Number of values in each block (could be None).
    # This is important for decompressing the images and
    # a description is available on page 171 of IMAP-Ultra Flight
    # Software Specification document (7523-9009_Rev_-.pdf).
    len_array: Union[
        int, None
    ]  # Length of the array to be decompressed (could be None).
    mantissa_bit_length: Union[int, None]  # used to determine the level of
    # precision that can be recovered from compressed data (could be None).


# Define PacketProperties instances directly in the module namespace
ULTRA_AUX = PacketProperties(
    apid=[880, 944],
    logical_source=["imap_ultra_l1a_45sensor-aux", "imap_ultra_l1a_90sensor-aux"],
    addition_to_logical_desc="Auxiliary",
    width=None,
    block=None,
    len_array=None,
    mantissa_bit_length=None,
)
ULTRA_RATES = PacketProperties(
    apid=[881, 945],
    logical_source=["imap_ultra_l1a_45sensor-rates", "imap_ultra_l1a_90sensor-rates"],
    addition_to_logical_desc="Image Rates",
    width=5,
    block=16,
    len_array=48,
    mantissa_bit_length=12,
)
ULTRA_TOF = PacketProperties(
    apid=[883, 947],
    logical_source=[
        "imap_ultra_l1a_45sensor-histogram",
        "imap_ultra_l1a_90sensor-histogram",
    ],
    addition_to_logical_desc="Time of Flight Images",
    width=4,
    block=15,
    len_array=None,
    mantissa_bit_length=4,
)
ULTRA_EVENTS = PacketProperties(
    apid=[896, 960],
    logical_source=["imap_ultra_l1a_45sensor-de", "imap_ultra_l1a_90sensor-de"],
    addition_to_logical_desc="Single Events",
    width=None,
    block=None,
    len_array=None,
    mantissa_bit_length=None,
)


# Module-level constant for event field ranges
# Module-level constant for event field ranges
EVENT_FIELD_RANGES = {
    # Coincidence Type
    "COIN_TYPE": (0, 2),
    # Start Type
    "START_TYPE": (2, 4),
    # Stop Type
    "STOP_TYPE": (4, 8),
    # Start Position Time to Digital Converter
    "START_POS_TDC": (8, 19),
    # Stop North Time to Digital Converter
    "STOP_NORTH_TDC": (19, 30),
    # Stop East Time to Digital Converter
    "STOP_EAST_TDC": (30, 41),
    # Stop South Time to Digital Converter
    "STOP_SOUTH_TDC": (41, 52),
    # Stop West Time to Digital Converter
    "STOP_WEST_TDC": (52, 63),
    # Coincidence North Time to Digital Converter
    "COIN_NORTH_TDC": (63, 74),
    # Coincidence South Time to Digital Converter
    "COIN_SOUTH_TDC": (74, 85),
    # Coincidence Discrete Time to Digital Converter
    "COIN_DISCRETE_TDC": (85, 96),
    # Energy/Pulse Height
    "ENERGY_PH": (96, 108),
    # Pulse Width
    "PULSE_WIDTH": (108, 119),
    # Event Flag Count
    "EVENT_FLAG_CNT": (119, 120),
    # Event Flag PHCmpSL
    "EVENT_FLAG_PHCMPSL": (120, 121),
    # Event Flag PHCmpSR
    "EVENT_FLAG_PHCMPSR": (121, 122),
    # Event Flag PHCmpCD
    "EVENT_FLAG_PHCMPCD": (122, 123),
    # Solid State Detector Flags
    "SSD_FLAG_7": (123, 124),
    "SSD_FLAG_6": (124, 125),
    "SSD_FLAG_5": (125, 126),
    "SSD_FLAG_4": (126, 127),
    "SSD_FLAG_3": (127, 128),
    "SSD_FLAG_2": (128, 129),
    "SSD_FLAG_1": (129, 130),
    "SSD_FLAG_0": (130, 131),
    # Constant Fraction Discriminator Flag Coincidence Top North
    "CFD_FLAG_COINTN": (131, 132),
    # Constant Fraction Discriminator Flag Coincidence Bottom North
    "CFD_FLAG_COINBN": (132, 133),
    # Constant Fraction Discriminator Flag Coincidence Top South
    "CFD_FLAG_COINTS": (133, 134),
    # Constant Fraction Discriminator Flag Coincidence Bottom South
    "CFD_FLAG_COINBS": (134, 135),
    # Constant Fraction Discriminator Flag Coincidence Discrete
    "CFD_FLAG_COIND": (135, 136),
    # Constant Fraction Discriminator Flag Start Right Full
    "CFD_FLAG_STARTRF": (136, 137),
    # Constant Fraction Discriminator Flag Start Left Full
    "CFD_FLAG_STARTLF": (137, 138),
    # Constant Fraction Discriminator Flag Start Position Right
    "CFD_FLAG_STARTRP": (138, 139),
    # Constant Fraction Discriminator Flag Start Position Left
    "CFD_FLAG_STARTLP": (139, 140),
    # Constant Fraction Discriminator Flag Stop Top North
    "CFD_FLAG_STOPTN": (140, 141),
    # Constant Fraction Discriminator Flag Stop Bottom North
    "CFD_FLAG_STOPBN": (141, 142),
    # Constant Fraction Discriminator Flag Stop Top East
    "CFD_FLAG_STOPTE": (142, 143),
    # Constant Fraction Discriminator Flag Stop Bottom East
    "CFD_FLAG_STOPBE": (143, 144),
    # Constant Fraction Discriminator Flag Stop Top South
    "CFD_FLAG_STOPTS": (144, 145),
    # Constant Fraction Discriminator Flag Stop Bottom South
    "CFD_FLAG_STOPBS": (145, 146),
    # Constant Fraction Discriminator Flag Stop Top West
    "CFD_FLAG_STOPTW": (146, 147),
    # Constant Fraction Discriminator Flag Stop Bottom West
    "CFD_FLAG_STOPBW": (147, 148),
    # Bin
    "BIN": (148, 156),
    # Phase Angle
    "PHASE_ANGLE": (156, 166),
}


RATES_KEYS = [
    # Start Right Full Constant Fraction Discriminator (CFD) Pulses
    "START_RF",
    # Start Left Full Constant Fraction Discriminator (CFD) Pulses
    "START_LF",
    # Start Position Right Full Constant Fraction Discriminator (CFD) Pulses
    "START_RP",
    # Start Position Left Constant Fraction Discriminator (CFD) Pulses
    "START_LP",
    # Stop Top North Constant Fraction Discriminator (CFD) Pulses
    "STOP_TN",
    # Stop Bottom North Constant Fraction Discriminator (CFD) Pulses
    "STOP_BN",
    # Stop Top East Constant Fraction Discriminator (CFD) Pulses
    "STOP_TE",
    # Stop Bottom East Constant Fraction Discriminator (CFD) Pulses
    "STOP_BE",
    # Stop Top South Constant Fraction Discriminator (CFD) Pulses
    "STOP_TS",
    # Stop Bottom South Constant Fraction Discriminator (CFD) Pulses
    "STOP_BS",
    # Stop Top West Constant Fraction Discriminator (CFD) Pulses
    "STOP_TW",
    # Stop Bottom West Constant Fraction Discriminator (CFD) Pulses
    "STOP_BW",
    # Coincidence Top North Constant Fraction Discriminator (CFD) Pulses
    "COIN_TN",
    # Coincidence Bottom North Constant Fraction Discriminator (CFD) Pulses
    "COIN_BN",
    # Coincidence Top South Constant Fraction Discriminator (CFD) Pulses
    "COIN_TS",
    # Coincidence Bottom South Constant Fraction Discriminator (CFD) Pulses
    "COIN_BS",
    # Coincidence Discrete Constant Fraction Discriminator (CFD) Pulses
    "COIN_D",
    # Solid State Detector (SSD) Energy Pulses
    "SSD0",
    "SSD1",
    "SSD2",
    "SSD3",
    "SSD4",
    "SSD5",
    "SSD6",
    "SSD7",
    # Start Position Time to Digital Converter (TDC) Chip VE Pulses
    "START_POS",
    # Stop North TDC-chip VE Pulses
    "STOP_N",
    # Stop East TDC-chip VE Pulses
    "STOP_E",
    # Stop South TDC-chip VE Pulses
    "STOP_S",
    # Stop West TDC-chip VE Pulses
    "STOP_W",
    # Coincidence North TDC-chip VE Pulses
    "COIN_N_TDC",
    # Coincidence Discrete TDC-chip VE Pulses
    "COIN_D_TDC",
    # Coincidence South TDC-chip VE Pulses
    "COIN_S_TDC",
    # Stop Top North Valid Pulse Height Flag
    "STOP_TOP_N",
    # Stop Bottom North Valid Pulse Height Flag
    "STOP_BOT_N",
    # Start-Right/Stop Single Coincidence.
    # Stop can be either Top or Bottom.
    # Coincidence is allowed, but not required.
    # No SSD.
    "START_RIGHT_STOP_COIN_SINGLE",
    # Start-Left/Stop Single Coincidence.
    # Stop can be either Top or Bottom.
    # Coincidence is allowed, but not required.
    # No SSD.
    "START_LEFT_STOP_COIN_SINGLE",
    # Start-Right/Stop/Coin Coincidence.
    # Double Coincidence.
    # Stop/Coin can be either Top or Bottom. No SSD.
    "START_RIGHT_STOP_COIN_DOUBLE",
    # Start-Left/Stop/Coin Coincidence.
    # Double Coincidence.
    # Stop/Coin can be either Top or Bottom. No SSD.
    "START_LEFT_STOP_COIN_DOUBLE",
    # Start/Stop/Coin Coincidence +
    # Position Match.
    # Double Coincidence + Fine Position Match
    # between Stop and Coin measurements.
    # No SSD.
    "START_STOP_COIN_POS",
    # Start-Right/SSD/Coin-D Coincidence.
    # Energy Coincidence.
    "START_RIGHT_SSD_COIN_D",
    # Start-Left/SSD/Coin-D Coincidence.
    # Energy Coincidence.
    "START_LEFT_SSD_COIN_D",
    # Event Analysis Activity Time.
    "EVENT_ACTIVE_TIME",
    # Events that would have been written to the FIFO.
    # (attempted to write).
    "FIFO_VALID_EVENTS",
    # Events generated by the pulser.
    "PULSER_EVENTS",
    # Coincidence (windowed) between the Stop/Coin top.
    "WINDOW_STOP_COIN",
    # Coincidence between Start Left and Window-Stop/Coin.
    "START_LEFT_WINDOW_STOP_COIN",
    # Coincidence between Start Right and Window-Stop/Coin.
    "START_RIGHT_WINDOW_STOP_COIN",
    # TODO: Below will be added later. It is not in the current data.
    # Processed events generated by the pulser.
    # "PROCESSED_PULSER_EVENTS",
    # Processed events.
    # "PROCESSED_EVENTS",
    # Discarded events.
    # "DISCARDED_EVENTS"
]


def append_fillval(decom_data: dict, packet):  # type: ignore[no-untyped-def]
    # ToDo, need packet param type
    """
    Append fill values to all fields.

    Parameters
    ----------
    decom_data : dict
        Parsed data.
    packet : space_packet_parser.packets.CCSDSPacket
        Packet.
    """
    for key in decom_data:
        if (key not in packet.header.keys()) and (key not in packet.user_data.keys()):
            decom_data[key].append(np.iinfo(np.int64).min)


def parse_event(event_binary: str) -> dict:
    """
    Parse a binary string representing a single event.

    Parameters
    ----------
    event_binary : str
        Event binary string.

    Returns
    -------
    fields_dict : dict
        Dict of the fields for a single event.
    """
    fields_dict = {}
    for field, (start, end) in EVENT_FIELD_RANGES.items():
        field_value = int(event_binary[start:end], 2)
        fields_dict[field] = field_value
    return fields_dict


def append_ccsds_fields(decom_data: dict, ccsds_data_object: object) -> None:
    """
    Append CCSDS fields to event_data.

    Parameters
    ----------
    decom_data : dict
        Parsed data.
    ccsds_data_object : DataclassInstance
        CCSDS data object.
    """
    for field in fields(ccsds_data_object.__class__):  # type: ignore[arg-type]
        ccsds_key = field.name
        if ccsds_key not in decom_data:
            decom_data[ccsds_key] = []
        decom_data[ccsds_key].append(getattr(ccsds_data_object, ccsds_key))
