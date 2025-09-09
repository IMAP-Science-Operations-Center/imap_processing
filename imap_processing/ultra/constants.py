"""Module for constants and useful shared classes used in Ultra."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class UltraConstants:
    """
    Constants for Ultra which can be used across different levels or classes.

    Attributes
    ----------
    D_SLIT_FOIL: float
        Shortest distance from slit to foil [mm]
    SLIT_Z: float
        Position of slit on Z axis [mm]
    YF_ESTIMATE_LEFT: float
        Front position of particle for left shutter [mm]
    YF_ESTIMATE_RIGHT: float
        Front position of particle for right shutter [mm]
    N_ELEMENTS: int
        Number of elements in lookup table
    TRIG_CONSTANT: float
        Trigonometric constant [mm]
    COMPOSITE_ENERGY_THRESHOLD: int
        DN threshold for composite energy
    Z_DSTOP: float
        Position of stop foil on Z axis [mm]
    Z_DS: float
        Position of slit on Z axis [mm]
    DF: float
        Distance from slit to foil [mm]
    DMIN: float
        Minimum distance between front and back detectors [mm]
    DMIN_SSD_CTOF: float
        SSD-specific correction to DMIN for time-of-flight normalization
    """

    D_SLIT_FOIL: float = 3.39
    SLIT_Z: float = 44.89
    YF_ESTIMATE_LEFT: float = 40.0
    YF_ESTIMATE_RIGHT: float = -40.0
    N_ELEMENTS: int = 256
    TRIG_CONSTANT: float = 81.92

    # Composite energy threshold for SSD events
    COMPOSITE_ENERGY_THRESHOLD: int = 1707

    # Geometry-related constants
    Z_DSTOP: float = 2.6 / 2  # Position of stop foil on Z axis [mm]
    Z_DS: float = 46.19 - (2.6 / 2)  # Position of slit on Z axis [mm]
    DF: float = 3.39  # Distance from slit to foil [mm]
    # Derived constants
    DMIN_PH_CTOF: float = (
        Z_DS - (2**0.5) * DF
    )  # Minimum distance between front and back detectors [mm]
    DMIN_SSD_CTOF: float = (DMIN_PH_CTOF**2) / (
        DMIN_PH_CTOF - Z_DSTOP
    )  # SSD-specific correction to DMIN [mm]

    # Conversion factors
    KEV_J = 1.602177e-16  # keV to joules
    J_KEV = 1 / KEV_J  # joules to keV
    MASS_H = 1.6735575e-27  # Mass of a hydrogen atom in kilograms.

    # Energy bin constants
    ALPHA = 0.2  # deltaE/E
    ENERGY_START = 3.385  # energy start for the Ultra grids
    N_BINS = 23  # number of energy bins

    # Constants for species determination based on ctof range.
    CTOF_SPECIES_MIN = 50
    CTOF_SPECIES_MAX = 200

    # RPMs for the Ultra instrument.
    # TODO: this is a placeholder.
    CULLING_RPM_MIN = 2.0
    CULLING_RPM_MAX = 6.0

    # Thresholds for culling based on counts (keV).
    CULLING_ENERGY_BIN_EDGES: ClassVar[list] = [
        3.0,
        10.0,
        20.0,
        50.0,
        300.0,
        1e5,
    ]
    PSET_ENERGY_BIN_EDGES: ClassVar[list] = [
        3.385,
        4.13722222222222,
        5.05660493827161,
        6.18029492455419,
        7.55369379667734,
        9.23229241816119,
        11.2839129555303,
        13.7914491678704,
        16.8562156496194,
        20.6020413495348,
        25.1802727605426,
        30.775888929552,
        37.6149753583414,
        45.9738587713061,
        56.1902718315964,
        68.6769989052845,
        83.93855421757,
        102.591566265919,
        125.38969210279,
        153.254068125632,
        187.310527709106,
        228.93508942224,
        279.809553738294,
        341.989454569026,
        1e5,
    ]

    # Valid event filter constants
    # Note these appear similar to image params constants
    # but they should be used only for the valid event filter.
    ETOFOFF1_EVENTFILTER = 100
    ETOFOFF2_EVENTFILTER = -50
    ETOFSLOPE1_EVENTFILTER = 6667
    ETOFSLOPE2_EVENTFILTER = 7500
    ETOFMAX_EVENTFILTER = 90
    ETOFMIN_EVENTFILTER = -400
    TOFDIFFTPMIN_EVENTFILTER = 226
    TOFDIFFTPMAX_EVENTFILTER = 266

    TOFXE_SPECIES_GROUPS: ClassVar[dict[str, list[int]]] = {
        "proton": [3],
        "non_proton": [20, 28, 36],
    }
    TOFXPH_SPECIES_GROUPS: ClassVar[dict[str, list[int]]] = {
        "proton": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "non_proton": [20, 21, 22, 23, 24, 25, 26],
    }
