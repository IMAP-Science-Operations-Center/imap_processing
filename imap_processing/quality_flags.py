"""Bitwise flagging."""

from enum import IntFlag


# TODO: as needed create a message for each name, value pair
#  for a more verbose description of what the flag indicates
class ImapQualityFlag(IntFlag):
    """Base Quality flags."""

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

        members = [member for member in BaseQualityFlags if member & self == member]
        return "|".join(str(m.name) for m in members if m != BaseQualityFlags.NONE)


class BaseQualityFlags(ImapQualityFlag):
    """Base Quality flags."""

    NONE = 0x0
    INF = 2**0  # bit 0, Infinite value
    MISSING_TELEM = 2**1  # bit 1, Missing telemetry
    NEG = 2**2  # bit 2, Negative value
    RES1 = 2**3  # bit 3, Reserved 1

    ALL = INF | MISSING_TELEM | NEG | RES1


class UltraQualityFlags(ImapQualityFlag):
    """Ultra specific Quality flags."""

    NONE = 0x0
    ULTRA_RES1 = 2**4  # bit 4, Ultra reserved 1
    ULTRA_RES2 = 2**5  # bit 5, Ultra reserved 2
    ULTRA_RES3 = 2**6  # bit 6, Ultra reserved 3

    ALL = ULTRA_RES1 | ULTRA_RES2 | ULTRA_RES3
