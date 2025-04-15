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

    # Thresholds for culling based on counts.
    CULLING_ENERGY_BIN_EDGES: ClassVar[list] = [0, 10, 20, 1e5]
