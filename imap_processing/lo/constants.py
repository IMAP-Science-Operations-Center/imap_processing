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
    PSET_PIVOT_ANGLE_TOLERANCE: float = 2.0

    # Ion species tracked. "H" is mandatory; any others for which we have histrates
    # may be added here.
    ELEMS = ("H", "O")

    # Hours into the day (UTC) for HK data to calculate median for pivot angle
    # estimation.
    PIVOT_HK_HOUR_RANGE: tuple[int, int] = (3, 15)

    N_CYCLE_SUM: int = 1  # Granularity of goodtime boundaries
    N_CYCLE_AVE: int = 7  # Cycles to average over when estimating background rates
    N_ESA_LEVELS: int = 7  # Total number of ESA levels
    N_SPINS_PER_ESA_LEVEL: int = 4  # Spins per ESA step within one histogram cycle

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

    # Maximum acceptable background count rates [counts/s]. There are separate
    # thresholds for RAM vs. anti-RAM, and for pivot near 90 deg vs. others.
    THRESHOLD_BG_RATE_RAM_90: float = 0.014
    THRESHOLD_BG_RATE_ANTI_RAM_90: float = 0.007
    THRESHOLD_BG_RATE_RAM_NON_90: float = 0.0175
    THRESHOLD_BG_RATE_ANTI_RAM_NON_90: float = 0.00875

    # Maximum time gap [s] between consecutive histogram epochs before treating them as
    # separate intervals.
    DELAY_MAX: int = 100
    # Pivot angles within this range [degrees] are treated as "near 90".
    PIVOT_90_RANGE: tuple[float, float] = 88.0, 92.0
    # Fraction of each cycle duration that contributes actual exposure.
    EXPOSURE_FACTOR: float = 0.5
    # Padding [s] added to begin/end of each goodtime interval to ensure complete
    # cycles are covered at interval edges.
    GOODTIME_PADDING: float = 2.0
