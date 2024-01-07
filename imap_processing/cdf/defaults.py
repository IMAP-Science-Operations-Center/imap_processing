"""Stores default values which can be used across instrument CDF files."""

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
