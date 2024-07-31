"""Bitwise flagging."""

from enum import IntFlag


class FlagNameMixin(IntFlag):
    """Modifies flags for Python versions < 3.11."""

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

        members = [member for member in self.__class__ if member & self == member]
        return "|".join(str(m).split(".", 1)[-1] for m in members if m != 0x0)


class CommonFlags(FlagNameMixin):
    """Common quality flags."""

    NONE = 0x0
    INF = 2**0  # bit 0, Infinite value
    NEG = 2**1  # bit 1, Negative value


class ENAFlags(FlagNameMixin):
    """Common ENA flags."""

    BADSPIN = 2**2  # bit 2, Bad spin


class ImapUltraFlags(FlagNameMixin):
    """IMAP Ultra flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    BADSPIN = ENAFlags.BADSPIN  # bit 2
    FLAG1 = 2**3  # bit 2


class ImapLoFlags(FlagNameMixin):
    """IMAP Lo flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    BADSPIN = ENAFlags.BADSPIN  # bit 2
    FLAG2 = 2**3  # bit 2


class HitFlags(
    FlagNameMixin,
):
    """Hit flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    FLAG3 = 2**2  # bit 2
