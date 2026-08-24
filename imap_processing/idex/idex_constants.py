"""Contains dataclasses to support IDEX processing."""

from dataclasses import dataclass
from enum import Enum, IntEnum

from imap_processing import imap_module_directory
from imap_processing.spice.geometry import SpiceFrame


class IDEXAPID(IntEnum):
    """Create ENUM for apid."""

    IDEX_SCIENCE = 1424
    IDEX_CATLST = 1419
    IDEX_EVT = 1418


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

# Low-rate timing constants
LOW_SAMPLE_RATE_HZ: float = 4.0625e6
SAMPLES_PER_BLOCK: int = 8
DT_BLOCK: float = SAMPLES_PER_BLOCK / LOW_SAMPLE_RATE_HZ

# Seconds in a day
SECONDS_IN_DAY = 86400
# Nanoseconds in day
NANOSECONDS_IN_DAY = SECONDS_IN_DAY * int(1e9)
# Picocoulombs to coulombs conversion factor
PICOCOULOMB_TO_COULOMB = 1e-12
# fg to kg conversion factor
FG_TO_KG = 1e-15

TARGET_HIGH_FREQUENCY_CUTOFF = 100

TARGET_NOISE_FREQUENCY = 7000

# This CSV was provided by the IDEX team.
# It defines the start and stop date of each 10-day window for IDEX l1a processing.
# All IDEX data will be grouped into these 10-day windows from l1a-l2a.
# the last window of each year may be less than 10 days. That is expected.
IDEX_10_DAY_RANGES_PATH = f"{imap_module_directory}/idex/idex_10_day_CDF_names.csv"


class ConversionFactors(float, Enum):
    """Conversion factors from DN to the engineering units for each waveform.

    TOF channels are reported in milliamperes (mA); target and ion-grid channels are
    reported in picocoulombs (pC).
    """

    TOF_High = 7.50e-5
    TOF_Low = 1.34e-1
    TOF_Mid = 2.93e-3
    Target_Low = 1.58e1
    Target_High = 1.63e-1
    Ion_Grid = 7.46e-4


SPICE_ARRAYS = [
    "ephemeris_position_x",
    "ephemeris_position_y",
    "ephemeris_position_z",
    "ephemeris_velocity_x",
    "ephemeris_velocity_y",
    "ephemeris_velocity_z",
    "longitude",
    "latitude",
    "solar_longitude",
    "spin_phase",
]

# Default IDEX Rectangular parameters
# Used in IDEX l2c processing
IDEX_SPACING_DEG = 6

# Define the pointing reference frame for IDEX
IDEX_EVENT_REFERENCE_FRAME = SpiceFrame.ECLIPJ2000
