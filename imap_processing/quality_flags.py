"""Bitwise flagging."""

from enum import IntFlag


class QualityFlags(IntFlag):
    """Quality flags."""

    # Base Quality flags
    NONE = 0x0
    INF = 2**0  # bit 0, Infinite value
    MISSING_TELEM = 2**1  # bit 1, Missing telemetry
    NEG = 2**2  # bit 2, Negative value
    RES1 = 2**3  # bit 3, Reserved 1

    # Ultra Quality Flags
    BAD_SPIN = 2**4  # bit 4, Ultra-specific flag
    FOV = 2**5  # bit 5, Ultra-specific flag

    @property
    def name(self) -> str:
        """
        Override the default name property to handle combined flags.

        Returns
        -------
        combined_name : str
            The combined name of the individual flags.
        """
        if self._name_ is not None:
            return self._name_

        members = [member for member in QualityFlags if member & self == member]
        return "|".join(
            str(m).split(".", 1)[-1] for m in members if m != QualityFlags.NONE
        )
