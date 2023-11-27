from collections import namedtuple
from dataclasses import dataclass

# Spacecraft clock time, a float divided into seconds and subseconds
TimeTuple = namedtuple("TimeTuple", "seconds subseconds")


@dataclass(frozen=True)
class GlowsConstants:
    """
    Constants for GLOWS which can be used across different levels or classes.

    Attributes
    ----------
    SUBSECOND_LIMIT: int
        subsecond limit for GLOWS clock (and consequently also onboard-interpolated
        IMAP clock)
    SCAN_CIRCLE_ANGULAR_RADIUS: float
        angular radius of IMAP/GLOWS scanning circle [deg]
    """

    SUBSECOND_LIMIT: int = 2_000_000
    SCAN_CIRCLE_ANGULAR_RADIUS: float = 75.0


@dataclass
class DirectEvent:
    """
    DirectEvent() class for IMAP/GLOWS.

    Authors: Marek Strumik, maro@cbk.waw.pl, Maxine Hartnett
    """

    timestamp: TimeTuple
    impulse_length: int
    multi_event: bool = False
