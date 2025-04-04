"""Collection of constant types or values for MAG."""

from enum import Enum

import numpy as np


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


class VecSec(Enum):
    """Enum for all valid vector rates (Vectors per second)."""

    ONE_VEC_PER_S = 1
    TWO_VECS_PER_S = 2
    FOUR_VECS_PER_S = 4
    EIGHT_VECS_PER_S = 8
    SIXTEEN_VECS_PER_S = 16
    THIRTY_TWO_VECS_PER_S = 32
    SIXTY_FOUR_VECS_PER_S = 64
    ONE_TWENTY_EIGHT_VECS_PER_S = 128


# Possible sensor rates
POSSIBLE_RATES = [e.value for e in VecSec]


class ModeFlags(Enum):
    """Enum for MAG mode flags: burst and normal (BURST + NORM)."""

    NORM = 0
    BURST = 1
    MISSING = -1


FIBONACCI_SEQUENCE = [
    1,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55,
    89,
    144,
    233,
    377,
    610,
    987,
    1597,
    2584,
    4181,
    6765,
    10946,
    17711,
    28657,
    46368,
    75025,
    121393,
    196418,
    317811,
    514229,
    832040,
    1346269,
    2178309,
    3524578,
    5702887,
    9227465,
    14930352,
    24157817,
    39088169,
    63245986,
    102334155,
    165580141,
]

MAX_FINE_TIME = np.iinfo(np.uint16).max  # maximum 16 bit unsigned int
AXIS_COUNT = 3
RANGE_BIT_WIDTH = 2
MAX_COMPRESSED_VECTOR_BITS = 60


def vectors_per_second_from_string(vecsec_string: str) -> dict:
    """
    Extract the vectors per second from a string into a dictionary.

    Dictionary format: {start_time: vecsec, start_time: vecsec}.

    Parameters
    ----------
    vecsec_string : str
        A string of the form "start:vecsec,start:vecsec" where start is the time in
        nanoseconds and vecsec is the number of vectors per second.

    Returns
    -------
    dict
        A dictionary of the form {start_time: vecsec, start_time: vecsec}.
    """
    vecsec_dict = {}
    vecsec_segments = vecsec_string.split(",")
    for vecsec_segment in vecsec_segments:
        start_time, vecsec = vecsec_segment.split(":")
        vecsec_dict[int(start_time)] = int(vecsec)

    return vecsec_dict
