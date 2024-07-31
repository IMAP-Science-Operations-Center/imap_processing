"""Bitwise flagging."""

from enum import IntFlag


class CommonFlags(IntFlag):
    """Common quality flags."""

    NONE = 0x0
    INF = 2**0  # bit 0, Infinite value
    NEG = 2**1  # bit 1, Negative value


class ENAFlags(IntFlag):
    """Common ENA flags."""

    BADSPIN = 2**2  # bit 2, Bad spin


class ImapUltraFlags(IntFlag):
    """IMAP Ultra flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    BADSPIN = ENAFlags.BADSPIN  # bit 2
    FLAG1 = 2**3  # bit 2


class ImapLoFlags(IntFlag):
    """IMAP Lo flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    BADSPIN = ENAFlags.BADSPIN  # bit 2
    FLAG2 = 2**3  # bit 2


class HitFlags(IntFlag):
    """Hit flags."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF  # bit 0
    NEG = CommonFlags.NEG  # bit 1
    FLAG3 = 2**2  # bit 2
