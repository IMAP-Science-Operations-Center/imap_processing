"""
Various classes and functions used throughout SWAPI processing.

This module contains utility classes and functions that are used by various
other SWAPI processing modules.
"""

from enum import IntEnum


class SWAPIAPID(IntEnum):
    """Create ENUM for apid."""

    SWP_HK = 1184
    SWP_SCI = 1188
    SWP_AUT = 1192


class SWAPIMODE(IntEnum):
    """Create ENUM for MODE."""

    LVENG = 0
    LVSCI = 1
    HVENG = 2
    HVSCI = 3
