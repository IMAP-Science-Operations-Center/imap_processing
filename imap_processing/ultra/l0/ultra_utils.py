"""Contains data classes to support Ultra L0 processing."""

from dataclasses import fields
from typing import NamedTuple, Union


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
    "cfd_flag_cointn": (131, 132),
    # Constant Fraction Discriminator Flag Coincidence Bottom North
    "cfd_flag_coinbn": (132, 133),
    # Constant Fraction Discriminator Flag Coincidence Top South
    "cfd_flag_coints": (133, 134),
    # Constant Fraction Discriminator Flag Coincidence Bottom South
    "cfd_flag_coinbs": (134, 135),
    # Constant Fraction Discriminator Flag Coincidence Discrete
    "cfd_flag_coind": (135, 136),
    # Constant Fraction Discriminator Flag Start Right Full
    "cfd_flag_startrf": (136, 137),
    # Constant Fraction Discriminator Flag Start Left Full
    "cfd_flag_startlf": (137, 138),
    # Constant Fraction Discriminator Flag Start Position Right
    "cfd_flag_startrp": (138, 139),
    # Constant Fraction Discriminator Flag Start Position Left
    "cfd_flag_startlp": (139, 140),
    # Constant Fraction Discriminator Flag Stop Top North
    "cfd_flag_stoptn": (140, 141),
    # Constant Fraction Discriminator Flag Stop Bottom North
    "cfd_flag_stopbn": (141, 142),
    # Constant Fraction Discriminator Flag Stop Top East
    "cfd_flag_stopte": (142, 143),
    # Constant Fraction Discriminator Flag Stop Bottom East
    "cfd_flag_stopbe": (143, 144),
    # Constant Fraction Discriminator Flag Stop Top South
    "cfd_flag_stopts": (144, 145),
    # Constant Fraction Discriminator Flag Stop Bottom South
    "cfd_flag_stopbs": (145, 146),
    # Constant Fraction Discriminator Flag Stop Top West
    "cfd_flag_stoptw": (146, 147),
    # Constant Fraction Discriminator Flag Stop Bottom West
    "cfd_flag_stopbw": (147, 148),
    # Bin
    "bin": (148, 156),
    # Phase Angle
    "phase_angle": (156, 166),
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
