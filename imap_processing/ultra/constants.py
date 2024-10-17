"""Module for constants and useful shared classes used in Ultra."""

from dataclasses import dataclass


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
    DMIN: float = (
        Z_DS - (2**0.5) * DF
    )  # Minimum distance between front and back detectors [mm]
    DMIN_SSD_CTOF: float = (DMIN**2) / (
        DMIN - Z_DSTOP
    )  # SSD-specific correction to DMIN [mm]

    # Conversion factors
    KEV_J = 1.602180000000000e-16#1.6021766339999e-16 # keV to joules
    MASS_H = 1.6735575e-27 # Mass of a hydrogen atom in kilograms.
