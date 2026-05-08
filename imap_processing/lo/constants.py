"""Constants for IMAP-Lo."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoConstants:
    """
    Constants for Lo which can be used across different levels.

    Attributes
    ----------
    PSET_PIVOT_ANGLE : float
        Expected pivot angle [degrees] for pointing sets for generating map products.
    PSET_PIVOT_ANGLE_TOLERANCE : float
        Absolute tolerance [degrees] for accepting a pset's pivot angle
        as sufficiently close to the required value.
    """

    PSET_PIVOT_ANGLE: float = 90.0
    PSET_PIVOT_ANGLE_TOLERANCE: float = 2.0
