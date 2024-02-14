"""Data class for Ultra Level 0 data."""
from enum import IntEnum


class Mode(IntEnum):  # TODO: Change name
    """Enum class for Ultra APIDs."""

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
    def __init__(self):
        """Define bit ranges for each field in the event packet."""
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

    def parse_event(self, event_binary):
        # Parse each field using its defined range and return a dictionary
        return {
            field: int(event_binary[start:end], 2)
            for field, (start, end) in self.field_ranges.items()
        }
