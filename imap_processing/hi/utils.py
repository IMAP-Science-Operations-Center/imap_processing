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
