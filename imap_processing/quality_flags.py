"""Bitwise flagging."""

from enum import IntFlag


class ModifyFlags(IntFlag):
    """
    Quality flags modified.

    This can be deleted once Python 3.11 is the
    last support version of Python.
    """

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

        members = [member for member in ModifyFlags if member & self == member]
        return "|".join(str(m).split(".", 1)[-1] for m in members if m != 0x0)


class CommonFlags(ModifyFlags):
    """Common quality flags."""

    NONE = 0x0
    INF = 2**0  # bit 0, Infinite value
    NEG = 2**1  # bit 1, Negative value


class ENAFlags(ModifyFlags):
    """Common ENA flags."""

    BADSPIN = 2**2  # bit 2, Bad spin


class ImapUltraFlags(ModifyFlags):
    """IMAP Ultra flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    BADSPIN = ENAFlags.BADSPIN  # bit 2
    FLAG1 = 2**3  # bit 2


class ImapLoFlags(ModifyFlags):
    """IMAP Lo flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    BADSPIN = ENAFlags.BADSPIN  # bit 2
    FLAG2 = 2**3  # bit 2


class HitFlags(ModifyFlags):
    """Hit flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    FLAG3 = 2**2  # bit 2
