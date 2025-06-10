"""Contains data classes to support Ultra L0 processing."""

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
        "imap_ultra_l1a_45sensor-histogram-ena-phxtof-hi-ang",
        "imap_ultra_l1a_90sensor-histogram-ena-phxtof-hi-ang",
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
ULTRA_HK = PacketProperties(
    apid=[
        866,
        867,
        869,
        870,
        873,
        874,
        876,
        877,
        930,
        931,
        933,
        934,
        937,
        938,
        940,
        941,
    ],
    logical_source=[
        "imap_ultra_l1a_45sensor-alarm",
        "imap_ultra_l1a_45sensor-memchecksum",
        "imap_ultra_l1a_45sensor-status",
        "imap_ultra_l1a_45sensor-bootstatus",
        "imap_ultra_l1a_45sensor-monitorlimits",
        "imap_ultra_l1a_45sensor-params",
        "imap_ultra_l1a_45sensor-scauto",
        "imap_ultra_l1a_45sensor-imgparams",
        "imap_ultra_l1a_90sensor-alarm",
        "imap_ultra_l1a_90sensor-memchecksum",
        "imap_ultra_l1a_90sensor-status",
        "imap_ultra_l1a_90sensor-bootstatus",
        "imap_ultra_l1a_90sensor-monitorlimits",
        "imap_ultra_l1a_90sensor-params",
        "imap_ultra_l1a_90sensor-scauto",
        "imap_ultra_l1a_90sensor-imgparams",
    ],
    addition_to_logical_desc="Housekeeping",
    width=None,
    block=None,
    len_array=None,
    mantissa_bit_length=None,
)
ULTRA_CMD_TEXT = PacketProperties(
    apid=[
        875,
        939,
    ],
    logical_source=[
        "imap_ultra_l1a_45sensor-cmdtext",
        "imap_ultra_l1a_90sensor-cmdtext",
    ],
    addition_to_logical_desc="Housekeeping with binary data",
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
    "start_rf",
    # Start Left Full Constant Fraction Discriminator (CFD) Pulses
    "start_lf",
    # Start Position Right Full Constant Fraction Discriminator (CFD) Pulses
    "start_rp",
    # Start Position Left Constant Fraction Discriminator (CFD) Pulses
    "start_lp",
    # Stop Top North Constant Fraction Discriminator (CFD) Pulses
    "stop_tn",
    # Stop Bottom North Constant Fraction Discriminator (CFD) Pulses
    "stop_bn",
    # Stop Top East Constant Fraction Discriminator (CFD) Pulses
    "stop_te",
    # Stop Bottom East Constant Fraction Discriminator (CFD) Pulses
    "stop_be",
    # Stop Top South Constant Fraction Discriminator (CFD) Pulses
    "stop_ts",
    # Stop Bottom South Constant Fraction Discriminator (CFD) Pulses
    "stop_bs",
    # Stop Top West Constant Fraction Discriminator (CFD) Pulses
    "stop_tw",
    # Stop Bottom West Constant Fraction Discriminator (CFD) Pulses
    "stop_bw",
    # Coincidence Top North Constant Fraction Discriminator (CFD) Pulses
    "coin_tn",
    # Coincidence Bottom North Constant Fraction Discriminator (CFD) Pulses
    "coin_bn",
    # Coincidence Top South Constant Fraction Discriminator (CFD) Pulses
    "coin_ts",
    # Coincidence Bottom South Constant Fraction Discriminator (CFD) Pulses
    "coin_bs",
    # Coincidence Discrete Constant Fraction Discriminator (CFD) Pulses
    "coin_d",
    # Solid State Detector (SSD) Energy Pulses
    "ssd0",
    "ssd1",
    "ssd2",
    "ssd3",
    "ssd4",
    "ssd5",
    "ssd6",
    "ssd7",
    # Start Position Time to Digital Converter (TDC) Chip VE Pulses
    "start_pos",
    # Stop North TDC-chip VE Pulses
    "stop_n",
    # Stop East TDC-chip VE Pulses
    "stop_e",
    # Stop South TDC-chip VE Pulses
    "stop_s",
    # Stop West TDC-chip VE Pulses
    "stop_w",
    # Coincidence North TDC-chip VE Pulses
    "coin_n_tdc",
    # Coincidence Discrete TDC-chip VE Pulses
    "coin_d_tdc",
    # Coincidence South TDC-chip VE Pulses
    "coin_s_tdc",
    # Stop Top North Valid Pulse Height Flag
    "stop_top_n",
    # Stop Bottom North Valid Pulse Height Flag
    "stop_bot_n",
    # Start-Right/Stop Single Coincidence.
    # Stop can be either Top or Bottom.
    # Coincidence is allowed, but not required.
    # No SSD.
    "start_right_stop_coin_single",
    # Start-Left/Stop Single Coincidence.
    # Stop can be either Top or Bottom.
    # Coincidence is allowed, but not required.
    # No SSD.
    "start_left_stop_coin_single",
    # Start-Right/Stop/Coin Coincidence.
    # Double Coincidence.
    # Stop/Coin can be either Top or Bottom. No SSD.
    "start_right_stop_coin_double",
    # Start-Left/Stop/Coin Coincidence.
    # Double Coincidence.
    # Stop/Coin can be either Top or Bottom. No SSD.
    "start_left_stop_coin_double",
    # Start/Stop/Coin Coincidence +
    # Position Match.
    # Double Coincidence + Fine Position Match
    # between Stop and Coin measurements.
    # No SSD.
    "start_stop_coin_pos",
    # Start-Right/SSD/Coin-D Coincidence.
    # Energy Coincidence.
    "start_right_ssd_coin_d",
    # Start-Left/SSD/Coin-D Coincidence.
    # Energy Coincidence.
    "start_left_ssd_coin_d",
    # Event Analysis Activity Time.
    "event_active_time",
    # Events that would have been written to the FIFO.
    # (attempted to write).
    "fifo_valid_events",
    # Events generated by the pulser.
    "pulser_events",
    # Coincidence (windowed) between the Stop/Coin top.
    "window_stop_coin",
    # Coincidence between Start Left and Window-Stop/Coin.
    "start_left_window_stop_coin",
    # Coincidence between Start Right and Window-Stop/Coin.
    "start_right_window_stop_coin",
    # TODO: Below will be added later. It is not in the current data.
    # Processed events generated by the pulser.
    # "processed_pulser_events",
    # Processed events.
    # "processed_events",
    # Discarded events.
    # "discarded_events"
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
