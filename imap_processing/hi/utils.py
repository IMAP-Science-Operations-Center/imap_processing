"""IMAP-Hi utils functions."""

from enum import IntEnum


class HIAPID(IntEnum):
    """Create ENUM for apid.

    Parameters
    ----------
    IntEnum : IntEnum
    """

    H45_APP_NHK = 754
    H45_SCI_CNT = 769
    H45_SCI_DE = 770

    H90_APP_NHK = 818
    H90_SCI_CNT = 833
    H90_SCI_DE = 834
