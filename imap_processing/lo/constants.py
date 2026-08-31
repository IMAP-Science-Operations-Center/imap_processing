"""Constants for IMAP-Lo."""

from dataclasses import dataclass
from typing import ClassVar, NamedTuple


class PivotAngleSpec(NamedTuple):
    """
    Pivot angle [degrees] and associated settings for a nominal pivot index.

    Attributes
    ----------
    pointing_index : int
        The unique pointing "index" (matching technical documentation for Imap-Lo)
    nominal : float
        Nominal pivot angle.
    min : float
        Lower bound of the acceptable pivot angle range.
    max : float
        Upper bound of the acceptable pivot angle range.
    bg_rate_ram : float, optional
        RAM background-rate threshold [counts/s] for this pivot angle. ``None``
        if no pivot-specific value is known, in which case
        ``LoConstants.THRESHOLD_BG_RATE_RAM_DEFAULT`` applies.
    bg_rate_anti_ram : float, optional
        Anti-RAM background-rate threshold [counts/s] for this pivot angle.
        ``None`` if no pivot-specific value is known, in which case
        ``LoConstants.THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT`` applies.
    """

    pointing_index: int
    nominal: float
    min: float
    max: float
    bg_rate_ram: float | None = None
    bg_rate_anti_ram: float | None = None


@dataclass(frozen=True)
class LoConstants:
    """Constants for Lo which can be used across different levels."""

    # Absolute tolerance [degrees] for accepting an input's pivot angle as
    # sufficiently close to the pivot angle of the map being made.
    PSET_PIVOT_ANGLE_TOLERANCE: float = 5.0

    # Empirical offset [degrees] added to the measured pivot angle when
    # projecting a look direction onto the RAM direction.
    PIVOT_RAM_OFFSET: float = 4.0

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

    # Bootstrap correction settings. The nominal coefficients of the ancillary
    # are scaled by BOOTSTRAP_SCALE before they are applied; the low and high
    # scalings bracket that choice and become the systematic error on the
    # corrected intensity.
    BOOTSTRAP_SCALE: float = 0.5
    BOOTSTRAP_SCALE_INTENSITY_HIGH: float = 0.25
    BOOTSTRAP_SCALE_INTENSITY_LOW: float = 1.0

    # The bootstrap correction of the highest ESA levels needs an ESA level
    # above them to subtract. That virtual "ESA 8" channel has no geometric
    # factor of its own; its intensity is extrapolated from the top two levels
    # with a power law, at this multiple of the top level's center energy.
    ESA_8_ENERGY_RATIO: float = 2.1
    # Width [pixels] of the neighborhood the spectral index of a pixel that has
    # no measurable one is taken from.
    BOOTSTRAP_SPECTRAL_INDEX_FILTER_SIZE: int = 3
    # The spectral index to extrapolate with when the map has none to offer.
    BOOTSTRAP_DEFAULT_SPECTRAL_INDEX: float = 1.6

    # Compton-Getting correction settings. The energy [eV] a hydrogen ENA has
    # in the spacecraft frame purely from the spacecraft's own motion, i.e.
    # 1/2 m_H U^2 at the nominal spacecraft speed of ~30 km/s. The kinematics
    # of the correction are scaled by it.
    CG_ENA_ENERGY_AT_SPACECRAFT_SPEED_EV: float = 4.661
    # The predictor-corrector that estimates the source spectrum behind the
    # observed one runs until the RMS change in the intensities falls below the
    # tolerance, or the iterations run out.
    CG_MAX_ITERATIONS: int = 20
    CG_CONVERGENCE_TOLERANCE: float = 0.005

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

    # Pivot angle specs keyed by nominal pivot angle. A measured pivot angle is
    # assigned to the first spec whose [min, max] range contains it; the ranges are
    # disjoint, so the ordering here does not matter.
    PIVOT_ANGLES: ClassVar[dict[int, PivotAngleSpec]] = {
        60: PivotAngleSpec(1, 60.0, 55.0, 65.0, None, None),
        75: PivotAngleSpec(2, 75.0, 70.0, 80.0, 0.035, 0.0175),
        90: PivotAngleSpec(3, 90.0, 85.0, 95.0, 0.028, 0.014),
        105: PivotAngleSpec(4, 105.0, 100.0, 110.0, 0.0224, 0.0112),
        120: PivotAngleSpec(5, 120.0, 115.0, 125.0, None, None),
        135: PivotAngleSpec(6, 135.0, 130.0, 140.0, None, None),
        148: PivotAngleSpec(7, 148.0, 143.0, 153.0, None, None),
        160: PivotAngleSpec(8, 160.0, 155.0, 165.0, None, None),
    }

    # Default background-rate thresholds [counts/s] when the pivot angle matches no
    # spec in PIVOT_ANGLES, or the matching spec has no pivot-specific value.
    # Currently set to nominal values for the 90-deg pivot angle.
    THRESHOLD_BG_RATE_RAM_DEFAULT: float = 0.028
    THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT: float = 0.014

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
