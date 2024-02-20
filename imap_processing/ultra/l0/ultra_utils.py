"""Data class for Ultra Level 0 data."""
from enum import IntEnum
from imap_processing.cdf.defaults import GlobalConstants


class ULTRAAPID(IntEnum):
    """Create ENUM for apid.

    Parameters
    ----------
    IntEnum : IntEnum
    """

    # Single Events: 0x380, 0x3c0
    ULTRA_EVENTS_45 = 896
    ULTRA_EVENTS_90 = 960

    # Auxiliary Data: 0x370, 0x3b0
    ULTRA_AUX_45 = 880
    ULTRA_AUX_90 = 994

    # Image Rates: 0x371, 0x3b1
    ULTRA_RATES_45 = 881
    ULTRA_RATES_90 = 945

    # TOF Images: 0x373,0x3b3
    ULTRA_TOF_45 = 883
    ULTRA_TOF_90 = 947


class EventParser:
    """Data class for parsing Ultra Level 0 event data from binary packets.

    Attributes
    ----------
    field_ranges : dict
        A dictionary mapping event field names to their respective start and end
        bit positions within a binary event string.
    scalar_fields : set
        A set containing the names of scalar fields that are relevant to the event data.

    Methods
    -------
    initialize_event_data(header: dict) -> dict:
        Initializes the data structure for storing event data.

    append_fillval(event_data: dict) -> None:
        Appends fill values to all event fields
        indicating missing or uninitialized data.

    parse_event(event_binary: str) -> dict:
        Parses a binary string representing a single event, extracting and converting
        event field values based on predefined field ranges.

    append_values(event_data: dict, packet: dict) -> None:
        Appends actual values to event fields and scalar fields for a
        given packet, updating the event data structure.
    """
    def __init__(self):
        self.field_ranges = {
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
        self.event_scalar_fields = {'SHCOARSE', 'SID', 'SPIN', 'ABORTFLAG', 'STARTDELAY', 'COUNT'}

    def initialize_event_data(self, header):
        """Initializes and returns the data structure for storing event data."""
        event_data = {field: [] for field in self.field_ranges}
        for scalar_field in self.event_scalar_fields:
            event_data[scalar_field] = []

        return event_data

    def append_fillval(self, event_data):
        """Appends fillvalue to all event fields except for specified scalar and CCSDS fields."""
        for key in event_data:
            if key not in self.event_scalar_fields:
                event_data[key].append(GlobalConstants.INT_FILLVAL)

    def parse_event(self, event_binary):
        """Parses a binary string representing a single event."""
        return {
            field: int(event_binary[start:end], 2)
            for field, (start, end) in self.field_ranges.items()
        }

    def append_values(self, event_data, packet):
        """Appends scalar fields to event_data."""
        for key in self.event_scalar_fields:
            event_data[key].append(packet.data[key].raw_value)