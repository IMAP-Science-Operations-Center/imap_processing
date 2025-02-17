"""Module for constants and useful shared classes used in GLOWS."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeTuple:
    """
    Spacecraft clock time, a float divided into seconds and subseconds.

    Attributes
    ----------
    seconds: int
        Seconds of clock, integer
    subseconds: int
        Subseconds of clock, defined as 1/SUB_SECOND_LIMIT th of a second. Will
        always be less than SUB_SECOND_LIMIT. If the class is initialized with a
        subsecond value above SUB_SECOND_LIMIT, the subseconds above the limit will be
        converted to seconds.
    """

    seconds: int
    subseconds: int

    def __post_init__(self) -> None:
        """Add any subseconds over the limit into the seconds field."""
        final_seconds = self.seconds
        final_subseconds = self.subseconds
        if final_subseconds >= GlowsConstants.SUBSECOND_LIMIT:
            final_seconds += self.subseconds // GlowsConstants.SUBSECOND_LIMIT
            final_subseconds = self.subseconds % GlowsConstants.SUBSECOND_LIMIT

        object.__setattr__(self, "seconds", final_seconds)
        object.__setattr__(self, "subseconds", final_subseconds)

    def to_seconds(self) -> np.double:
        """
        Convert the TimeTuple to seconds.

        Returns
        -------
        np.single
            TimeTuple in seconds.
        """
        return np.double(
            self.seconds + self.subseconds / GlowsConstants.SUBSECOND_LIMIT
        )


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

    Attributes
    ----------
    timestamp: TimeTuple
        Timestamp for the direct event
    impulse_length: int
        Direct event data
    multi_event: bool
        If the event was a multi event. Defaults to False.
    """

    timestamp: TimeTuple
    impulse_length: int
    multi_event: bool = False

    def to_list(self) -> list:
        """
        Convert object to list [seconds, subseconds, impulse length, multievent].

        Returns
        -------
        list
            Converted object to list.
        """
        return [
            self.timestamp.seconds,
            self.timestamp.subseconds,
            self.impulse_length,
            self.multi_event,
        ]
