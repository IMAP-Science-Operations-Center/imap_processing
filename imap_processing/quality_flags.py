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


class ImapHkUltraFlags(FlagNameMixin):
    """IMAP Ultra flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    BADSPIN = ENAFlags.BADSPIN  # bit 2
    FLAG1 = 2**3  # bit 3


class ImapAttitudeUltraFlags(FlagNameMixin):
    """IMAP Ultra flags."""

    NONE = CommonFlags.NONE
    SPINRATE = 2**0  # bit 0
    AUXMISMATCH = 2**1  # bit 1 # aux packet does not match Universal Spin Table


class ImapRatesUltraFlags(FlagNameMixin):
    """IMAP Ultra Rates flags."""

    NONE = CommonFlags.NONE
    ZEROCOUNTS = 2**0  # bit 0
    HIGHRATES = 2**1  # bit 1


class ImapLoFlags(FlagNameMixin):
    """IMAP Lo flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    BADSPIN = ENAFlags.BADSPIN  # bit 2
    FLAG2 = 2**3  # bit 3


class HitFlags(
    FlagNameMixin,
):
    """Hit flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    FLAG3 = 2**2  # bit 2


class SWAPIFlags(
    FlagNameMixin,
):
    """SWAPI flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    SWP_PCEM_COMP = 2**2  # bit 2
    SWP_SCEM_COMP = 2**3  # bit 3
    SWP_COIN_COMP = 2**4  # bit 4
    OVR_T_ST = 2**5  # bit 5
    UND_T_ST = 2**6  # bit 6
    PCEM_CNT_ST = 2**7  # bit 7
    SCEM_CNT_ST = 2**8  # bit 8
    PCEM_V_ST = 2**9  # bit 9
    PCEM_I_ST = 2**10  # bit 10
    PCEM_INT_ST = 2**11  # bit 11
    SCEM_V_ST = 2**12  # bit 12
    SCEM_I_ST = 2**13  # bit 13
    SCEM_INT_ST = 2**14  # bit 14
