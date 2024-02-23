"""Contains data classes to support Ultra L0 processing."""
from dataclasses import fields
from typing import NamedTuple

from imap_processing.cdf.defaults import GlobalConstants


class PacketProperties(NamedTuple):
    """Class that represents properties of the ULTRA packet type."""

    apid: int
    width: int
    block: int
    len_array: int
    mantissa_bit_length: int


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
    "coin_type": (0, 2),
    "start_type": (2, 4),
    "stop_type": (4, 8),
    "start_pos_tdc": (8, 19),
    "stop_north_tdc": (19, 30),
    "stop_east_tdc": (30, 41),
    "stop_south_tdc": (41, 52),
    "stop_west_tdc": (52, 63),
    "coin_north_tdc": (63, 74),
    "coin_south_tdc": (74, 85),
    "coin_discrete_tdc": (85, 96),
    "energy_ph": (96, 108),
    "pulse_width": (108, 119),
    "event_flag_cnt": (119, 120),
    "event_flag_phcmpsl": (120, 121),
    "event_flag_phcmpsr": (121, 122),
    "event_flag_phcmpcd": (122, 123),
    "ssd_flag_7": (123, 124),
    "ssd_flag_6": (124, 125),
    "ssd_flag_5": (125, 126),
    "ssd_flag_4": (126, 127),
    "ssd_flag_3": (127, 128),
    "ssd_flag_2": (128, 129),
    "ssd_flag_1": (129, 130),
    "ssd_flag_0": (130, 131),
    "cfd_flag_cointn": (131, 132),
    "cfd_flag_coinbn": (132, 133),
    "cfd_flag_coints": (133, 134),
    "cfd_flag_coinbs": (134, 135),
    "cfd_flag_coind": (135, 136),
    "cfd_flag_startrf": (136, 137),
    "cfd_flag_startlf": (137, 138),
    "cfd_flag_startrp": (138, 139),
    "cfd_flag_startlp": (139, 140),
    "cfd_flag_stoptn": (140, 141),
    "cfd_flag_stopbn": (141, 142),
    "cfd_flag_stopte": (142, 143),
    "cfd_flag_stopbe": (143, 144),
    "cfd_flag_stopts": (144, 145),
    "cfd_flag_stopbs": (145, 146),
    "cfd_flag_stoptw": (146, 147),
    "cfd_flag_stopbw": (147, 148),
    "bin": (148, 156),
    "phase_angle": (156, 166),
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
