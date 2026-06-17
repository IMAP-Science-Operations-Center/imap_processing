"""Constants for IMAP-Lo."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LoConstants:
    """Constants for Lo which can be used across different levels."""

    # Expected pivot angle [degrees] for pointing sets for generating map products.
    PSET_PIVOT_ANGLE: float = 90.0
    # Absolute tolerance [degrees] for accepting a pset's pivot angle
    # as sufficiently close to the required value.
    PSET_PIVOT_ANGLE_TOLERANCE: float = 45.0

    # Ion species tracked. "H" is mandatory (and should be the first element);
    # any others for which we have histrates may be added here.
    ELEMS = ("H", "O")

    # Hours into the day (UTC) for HK data to calculate median for pivot angle
    # estimation.
    PIVOT_HK_HOUR_RANGE: tuple[float, float] = (0.5, 22.5)

    N_CYCLE_SUM: int = 1  # Granularity of goodtime boundaries
    N_CYCLE_AVE: int = 7  # Cycles to average over when estimating background rates
    N_ESA_LEVELS: int = 7  # Total number of ESA levels
    N_SPINS_PER_ESA_LEVEL: int = 4  # Spins per ESA step within one histogram cycle
    N_SPIN_ANGLE_BINS: int = 60  # Number of angular bins within a spin

    # Nominal spin period [s]. True spin duration is NOT 15 seconds.
    NOMINAL_SPIN_PERIOD_SEC: float = 15.0

    # One histogram accumulation cycle duration [s]
    HISTOGRAM_CYCLE_EPOCHS: int = (
        N_ESA_LEVELS * N_SPINS_PER_ESA_LEVEL * int(NOMINAL_SPIN_PERIOD_SEC)
    )
    RAM_ESA_LEVELS: tuple[int, ...] = (
        6,
        7,
    )  # ESA levels for RAM estimation (1-indexed)

    # Histogram angular bins (0-indexed) corresponding to the RAM and anti-RAM look
    # directions
    RAM_HISTOGRAM_BINS: tuple[slice, ...] = (slice(0, 20), slice(50, 60))
    ANTI_RAM_HISTOGRAM_BINS: tuple[slice, ...] = (slice(20, 50),)

    # Nominal background rates [counts/s] for each species
    BG_RATES: ClassVar[dict[str, float]] = {"H": 0.0014925, "O": 0.000136635}
    # When no exposure is available, scale the nominal rate down as a conservative
    # estimate.
    BG_RATE_FALLBACK_SCALE: ClassVar[dict[str, float]] = {"H": 1.0, "O": 0.3}
    # Minimum non-zero background rate floor = nominal / divisor
    BG_RATE_FLOOR_DIVISOR: ClassVar[dict[str, float]] = {"H": 50.0, "O": 150.0}

    # Background-rate thresholds [counts/s] by pivot-angle range (low, high) [deg].
    # Each value is (ram_threshold, anti_ram_threshold).
    # The first matching open interval (low < pivot < high) is used; if none matches,
    # THRESHOLD_BG_RATE_RAM_DEFAULT / THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT apply.
    PIVOT_ANGLE_THRESHOLDS: ClassVar[dict[tuple[float, float], tuple[float, float]]] = {
        (88.0, 92.0): (0.028, 0.014),
        (73.0, 77.0): (0.035, 0.0175),
        (103.0, 107.0): (0.0224, 0.0112),
    }

    # Default background-rate thresholds [counts/s] when no pivot range matches.
    THRESHOLD_BG_RATE_RAM_DEFAULT: float = 0.0175
    THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT: float = 0.00875

    # Maximum time gap [s] between consecutive histogram epochs before treating them as
    # separate intervals.
    DELAY_MAX: int = 100
    # Fraction of each cycle duration that contributes actual exposure.
    EXPOSURE_FACTOR: float = 0.5
    # Padding [s] added to begin/end of each goodtime interval to ensure complete
    # cycles are covered at interval edges.
    GOODTIME_PADDING: float = 2.0

    # Star-sensor spin-angle binning offset (fractional bin-index shift used when
    # computing sample centers), keyed by the IFB star-sync housekeeping state
    # (ifb_ctrl_star_sync). Flight software 4.8 enabled star sync ("EN"),
    # switching from binning to the bin center (+0.5) to the left edge (+0.0).
    STAR_BIN_OFFSET_BY_SYNC: ClassVar[dict[str | None, float]] = {
        "DS": 0.5,  # star sync disabled (pre FSW 4.8)
        "EN": 0.0,  # star sync enabled (FSW 4.8+)
    }

    # Number of ending bins to exclude from each star-sensor profile average.
    STAR_END_BINS_TO_EXCLUDE: int = 2
    # Minimum COUNT value for a star-sensor record to be considered valid.
    STAR_MIN_COUNT_THRESHOLD: int = 700
