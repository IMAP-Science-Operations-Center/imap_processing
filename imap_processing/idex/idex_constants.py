"""Contains dataclasses to support IDEX processing."""

from dataclasses import dataclass


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
