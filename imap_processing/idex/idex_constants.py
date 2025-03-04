"""Contains dataclasses to support IDEX processing."""

from dataclasses import dataclass
from enum import Enum


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


# FM sampling rate (quartz oscillator)
# Seconds per sample.
FM_SAMPLING_RATE = 0.0038466235767167234e-6
# Nanoseconds to seconds conversion
NS_TO_S = 1e-9
# Microseconds to seconds conversion
US_TO_S = 1e-6

TARGET_HIGH_FREQUENCY_CUTOFF = 100

TARGET_NOISE_FREQUENCY = 7000


class ConversionFactors(float, Enum):
    """Conversion factor values (DN to picocoulombs) for each of the six waveforms."""

    TOF_High = 2.89e-4
    TOF_Low = 5.14e-4
    TOF_Mid = 1.13e-2
    Target_Low = 1.58e1
    Target_High = 1.63e-1
    Ion_Grid = 7.46e-4
