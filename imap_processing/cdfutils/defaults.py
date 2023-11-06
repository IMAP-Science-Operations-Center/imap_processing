from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalConstants:
    """
    Class for shared constants across CDF classes.

    Attributes
    ----------
    INT_FILLVAL: np.int64
        Recommended FILLVAL for all integers (numpy int64 min)
    INT_MAXVAL: np.int64
        Recommended maximum value for INTs (numpy int64 max)
    DOUBLE_FILLVAL: np.float64
        Recommended FILLVALL for all floats
    MIN_EPOCH: int
        Recommended minimum epoch based on MMS approved values
    MAX_EPOCH: int
        Recommended maximum epoch based on MMS approved values
    """

    INT_FILLVAL = np.iinfo(np.int64).min
    INT_MAXVAL = np.iinfo(np.int64).max
    DOUBLE_FILLVAL = np.float64(-1.0e31)
    MIN_EPOCH = -315575942816000000
    MAX_EPOCH = 946728069183000000


@dataclass
class IdexConstants:
    """
    Class for IDEX constants.

    Attributes
    ----------
    DATA_MIN: int = 0
        Data is in a 12 bit unsigned INT. It could go down to 0 in theory
    DATA_MAX: int = 4096
        Data is in a 12 bit unsigned INT. It cannot exceed 4096 (2^12)
    SAMPLE_RATE_MIN: int = -130
        The minimum sample rate, all might be negative
    SAMPLE_RATE_MAX: int = 130
        The maximum sample rate. Samples span 130 microseconds at the most, and all
        might be positive
    """

    DATA_MIN: int = 0
    DATA_MAX: int = 4096
    SAMPLE_RATE_MIN: int = -130
    SAMPLE_RATE_MAX: int = 130
