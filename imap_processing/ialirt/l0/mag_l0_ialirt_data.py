"""Classes for Level 0 MAG I-ALiRT data."""

from __future__ import annotations


class Packet0:
    """
    Class for packet 0.

    Parameters
    ----------
    status : int
        24-bit integer value.

    Notes
    -----
    Bits 23-22 → Packet Number (2-bit value)
    Bits 21-17 → hk1v5_warn, hk1v5_danger, hk1v5c_warn, hk1v5c_danger,
    hk1v8_warn (5-bit value)
    Bits 16-13 → hk1v8_danger, hk1v8c_warn, hk1v8c_danger (4-bit value)
    Bit  5 → fob_saturated (1-bit value)
    Bit  4 → fib_saturated (1-bit value)
    Bits 3-0 → mode (4-bit value)
    Bits 12-6 → icu_temp (7-bit value)
    """

    def __init__(self, status: int) -> None:
        self.hk1v5_warn = (status >> 20) & 0x01
        self.hk1v5_danger = (status >> 19) & 0x01
        self.hk1v5c_warn = (status >> 18) & 0x01
        self.hk1v5c_danger = (status >> 17) & 0x01
        self.hk1v8_warn = (status >> 16) & 0x01
        self.hk1v8_danger = (status >> 15) & 0x01
        self.hk1v8c_warn = (status >> 14) & 0x01
        self.hk1v8c_danger = (status >> 13) & 0x01
        self.fob_saturated = (status >> 5) & 0x01
        self.fib_saturated = (status >> 4) & 0x01
        self.mode = (status >> 0) & 0x0F
        # instrument control unit temperature
        self.icu_temp = ((status >> 6) & 0x7F) << 5


class Packet1:
    """
    Class for packet 1.

    Parameters
    ----------
    status : int
        24-bit integer value.

    Notes
    -----
    Bit  5 → pri_isvalid (1-bit value)
    Bits 4-1 → hk2v5_warn, hk2v5_danger, hk2v5c_warn, hk2v5c_danger (4-bit value)
    Bits 16-9 → hk3v3 (8-bit value)
    Bits 8-0 → hk3v3_current (9-bit value)
    """

    def __init__(self, status: int) -> None:
        self.hk2v5_warn = (status >> 20) & 0x01
        self.hk2v5_danger = (status >> 19) & 0x01
        self.hk2v5c_warn = (status >> 18) & 0x01
        self.hk2v5c_danger = (status >> 17) & 0x01
        self.hk3v3 = ((status >> 9) & 0xFF) << 4
        self.hk3v3_current = ((status >> 0) & 0x1FF) << 3
        # Primary sensor validity flag
        self.pri_isvalid = (status >> 21) & 0x01


class Packet2:
    """
    Class for packet 2.

    Parameters
    ----------
    status : int
        24-bit integer value.

    Notes
    -----
    Bits 23-22 → Packet Number (2-bit value)
    Bits 21-17 → Various warning/danger flags (5-bit value)
    Bits 16-9 → hkn8v5 (8-bit value)
    Bits 8-0 → hkn8v5_current (9-bit value)
    """

    def __init__(self, status: int) -> None:
        self.hkp8v5_warn = (status >> 20) & 0x01
        self.hkp8v5_danger = (status >> 19) & 0x01
        self.hkp8v5c_warn = (status >> 18) & 0x01
        self.hkp8v5c_danger = (status >> 17) & 0x01
        self.hkn8v5 = ((status >> 9) & 0xFF) << 4
        self.hkn8v5_current = ((status >> 0) & 0x1FF) << 3


class Packet3:
    """
    Class for packet 3.

    Parameters
    ----------
    status : int
        24-bit integer value.

    Notes
    -----
    Bits 20-13 → fob_temp (8-bit value, shifted left by 4)
    Bits 12-5 → fib_temp (8-bit value, shifted left by 4)
    Bits 4-3 → fob_range (2-bit value)
    Bits 2-1 → fib_range (2-bit value)
    Bit 0 → multbit_errs (1-bit value)
    Bit 5 → sec_isvalid (1-bit value, overlapping with fib_temp extraction)
    """

    def __init__(self, status: int) -> None:
        # Temp of the Front End Electronics (FEE) connected to MAGo
        self.fob_temp = ((status >> 13) & 0xFF) << 4
        # Temp of the Front End Electronics (FEE) connected to MAGi
        self.fib_temp = ((status >> 5) & 0xFF) << 4
        self.fob_range = (status >> 3) & 0x03
        self.fib_range = (status >> 1) & 0x03
        self.multbit_errs = (status >> 0) & 0x01
        # Secondary sensor validity flag
        self.sec_isvalid = (status >> 21) & 0x01
