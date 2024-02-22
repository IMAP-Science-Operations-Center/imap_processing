"""Contains data classes to support Ultra L0 processing."""
from dataclasses import fields
from enum import Enum
from typing import NamedTuple

from imap_processing.cdf.defaults import GlobalConstants


class PacketProperties(NamedTuple):
    """Class that represents properties of the ULTRA packet type."""

    apid: int
    width: int
    block: int
    len_array: int
    mantissa_bit_length: int


class UltraParams(Enum):
    """Enumerated packet properties for ULTRA."""

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


class ParserHelper:
    """
    A utility class for parsing Ultra Level 0 data.

    Attributes
    ----------
    event_field_ranges : dict
        Maps event field names to tuples indicating their
        start and end bit positions in a binary event string.

    Methods
    -------
    initialize_event_data() -> dict:
        Creates and returns a dictionary with keys for
        each event field and empty lists as values.

    append_fillval(decom_data: dict, packet) -> None:
        Inserts fill values into `decom_data` for any fields not
        present, indicating missing data.

    parse_event(event_binary: str) -> dict:
        Converts a binary string `event_binary` into a
        dictionary of event fields and values.

    append_ccsds_fields(decom_data: dict, ccsds_data_object) -> None:
        Adds data from a CCSDS data object to `decom_data`.
    """

    def __init__(self):
        self.event_field_ranges = {
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
            "event_flags": (119, 123),
            "ssd_flags": (123, 131),
            "cfd_flags": (131, 148),
            "bin": (148, 156),
            "phase_angle": (156, 166),
        }

    def initialize_event_data(self, decom_data):
        """Initialize the decom_data dictionary with empty lists for new keys.

        Adds new keys with empty lists for those not already present.
        """
        for field in self.event_field_ranges:
            if field not in decom_data:
                decom_data[field] = []
        return decom_data

    def append_fillval(self, decom_data, packet):
        """Append fill values to all fields."""
        for key in decom_data:
            if (key not in packet.header.keys()) and (key not in packet.data.keys()):
                decom_data[key].append(GlobalConstants.INT_FILLVAL)

    def parse_event(self, event_binary):
        """Parse a binary string representing a single event."""
        return {
            # 2 is the base for binary conversion
            field: int(event_binary[start:end], 2)
            for field, (start, end) in self.event_field_ranges.items()
        }

    def append_ccsds_fields(self, decom_data, ccsds_data_object):
        """Append ccsds fields to event_data."""
        for field in fields(ccsds_data_object.__class__):
            ccsds_key = field.name
            if ccsds_key not in decom_data:
                decom_data[ccsds_key] = []
            decom_data[ccsds_key].append(getattr(ccsds_data_object, ccsds_key))
