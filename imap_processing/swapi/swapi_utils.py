"""
Various classes and functions used throughout SWAPI processing.

This module contains utility classes and functions that are used by various
other SWAPI processing modules.
"""

from enum import Enum, IntEnum


class SWAPIAPID(IntEnum):
    """Create ENUM for apid."""

    SWP_HK = 1184
    SWP_SCI = 1188
    SWP_AUT = 1192


class SWAPIMODE(Enum):
    """Create ENUM for MODE."""

    LVENG = "LVENG"
    LVSCI = "LVSCI"
    HVENG = "HVENG"
    HVSCI = "HVSCI"
