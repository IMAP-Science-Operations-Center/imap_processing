"""IMAP-Lo APIDs ENUM."""

from enum import IntEnum


class LoAPID(IntEnum):
    """IMAP-Lo APIDs."""

    ILO_AUTO = 672  # Autonomy
    ILO_BOOT_HK = 673  # Boot memory dump
    ILO_BOOT_MEMDMP = 674  # Boot housekeeping
    ILO_APP_SHK = 676  # Static housekeeping (values that don't change)
    ILO_APP_NHK = 677  # Nominal housekeeping (engineering, health and status)
    ILO_EVTMSG = 678  # Event message
    ILO_MEMDMP = 679  # App memory dump
    ILO_RAW_CNT = 689  # Raw counter values, intended for use in engineering modes
    ILO_RAW_DE = 690  # Raw direct events, intended for use in engineering modes
    ILO_RAW_STAR = 691  # Raw star sensor, intended for use in HVENG, every spin
    ILO_SCI_CNT = 705  # Science rates, including derived and histograms
    ILO_SCI_DE = 706  # Science direct event data
    ILO_STAR = 707  # Science star sensor data, every spin
    ILO_SPIN = 708  # Spin information for each science cycle (28 spins)
    ILO_DIAG_CDH = 721  # Diagnostic CDH (raw register dumps)
    ILO_DIAG_IFB = 722  # Diagnostic IFB (raw register dumps)
    ILO_DIAG_TOF_BD = 723  # Diagnostic TOF Board (raw register dumps)
    ILO_DIAG_BULK_HVPS = 724  # Diagnostic bulk HVPS (raw register dumps)
    ILO_DIAG_PCC = 725  # Diagnostic PCC (raw register dumps)
