"""Bitwise flagging."""

from enum import IntFlag


class BaseQualityFlags(IntFlag):
    """Base Quality flags."""

    NONE = 0x0
    BASE_INF = 0x1  # bit 0, Infinite value
    BASE_MISSING_TELEM = 0x2  # bit 1, Missing telemetry
    BASE_NEG = 0x4  # bit 2, Negative value
    BASE_RES1 = 0x8  # bit 3, Reserved 1

    BASE_ALL = BASE_INF | BASE_MISSING_TELEM | BASE_NEG | BASE_RES1

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


class UltraQualityFlags(IntFlag):
    """Ultra specific Quality flags."""

    NONE = 0x0
    ULTRA_RES1 = 0x10  # bit 4, Ultra reserved 1
    ULTRA_RES2 = 0x20  # bit 5, Ultra reserved 2
    ULTRA_RES3 = 0x40  # bit 6, Ultra reserved 3

    ALL = ULTRA_RES1 | ULTRA_RES2 | ULTRA_RES3

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

        members = [member for member in UltraQualityFlags if member & self == member]
        return "|".join(str(m.name) for m in members if m != UltraQualityFlags.NONE)
