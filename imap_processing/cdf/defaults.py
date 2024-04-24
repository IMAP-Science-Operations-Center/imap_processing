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
    FLOAT_MAXVAL = np.finfo(np.float64).max
    DOUBLE_FILLVAL = np.float64(-1.0e31)
    # 1900-01-01T00:00:00
    MIN_EPOCH = -315575942816000000
    # 2100-01-01T00:00:00
    MAX_EPOCH = 3155630469184000000
    # 32-bits max int value
    UINT32_MAXVAL = np.iinfo(np.uint32).max
    UINT16_MAXVAL = np.iinfo(np.uint16).max
