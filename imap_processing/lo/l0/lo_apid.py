"""IMAP-Lo APIDs ENUM."""

from enum import IntEnum


class LoAPID(IntEnum):
    """IMAP-Lo APIDs."""

    ILO_BOOT_HK = 673  # Boot memory dump
    ILO_APP_SHK = 676  # Static housekeeping (values that don't change)
    ILO_APP_NHK = 677  # Nominal housekeeping (engineering, health and status)
    ILO_SCI_CNT = 705  # Science rates, including derived and histograms
    ILO_SCI_DE = 706  # Science direct event data
    ILO_STAR = 707  # Science star sensor data, every spin
    ILO_SPIN = 708  # Spin information for each science cycle (28 spins)
