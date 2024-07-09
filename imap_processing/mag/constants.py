"""Collection of constant types or values for MAG."""

from enum import Enum


class DataMode(Enum):
    """Enum for MAG data modes: burst and normal (BURST + NORM)."""

    BURST = "BURST"
    NORM = "NORM"


class Sensor(Enum):
    """Enum for MAG sensors: raw, MAGo, and MAGi (RAW, MAGO, MAGI)."""

    MAGO = "MAGO"
    MAGI = "MAGI"
    RAW = "RAW"
