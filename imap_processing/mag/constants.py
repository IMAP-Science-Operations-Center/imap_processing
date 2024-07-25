"""Collection of constant types or values for MAG."""

from enum import Enum


class DataMode(Enum):
    """
    Enum for MAG data modes: burst and normal (BURST + NORM).

    Attributes
    ----------
    BURST: str
        Burst data mode - higher frequency data
    NORM: str
        Normal data mode - lower frequency data (downsampled from burst)
    """

    BURST = "BURST"
    NORM = "NORM"


class Sensor(Enum):
    """
    Enum for MAG sensors: raw, MAGo, and MAGi (RAW, MAGO, MAGI).

    Attributes
    ----------
    MAGO : str
        MAGo sensor - for the outboard sensor. This is nominally expected to be the
        primary sensor.
    MAGI : str
        MAGi sensor - for the inboard sensor.
    RAW : str
        RAW data - contains both sensors. Here, the vectors are unprocessed.
    """

    MAGO = "MAGO"
    MAGI = "MAGI"
    RAW = "RAW"


class PrimarySensor(Enum):
    """
    Enum for primary sensor: MAGo and MAGi (MAGO, MAGI).

    This corresponds to the PRI_SENS field in the MAG Level 0 data.

    Attributes
    ----------
    MAGO : int
        Primary sensor is MAGo.
    MAGI : int
        Primary sensor is MAGi.
    """

    MAGO = 0
    MAGI = 1
