"""Dataclasses for Level 0 MAG I-ALiRT data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Packet0:
    """Dataclass for packet 0."""

    hk1v5_warn: int
    hk1v5_danger: int
    hk1v5c_warn: int
    hk1v5c_danger: int
    hk1v8_warn: int
    hk1v8_danger: int
    hk1v8c_warn: int
    hk1v8c_danger: int
    fob_saturated: int
    fib_saturated: int
    mode: int
    icu_temp: int


def decode_packet0(status: int) -> Packet0:
    """
    Decode packet 0.

    Parameters
    ----------
    status : int
        24-bit integer value.

    Returns
    -------
    list[int]
        List of extracted bytes.

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
    hk1v5_warn = (status >> 20) & 0x01
    hk1v5_danger = (status >> 19) & 0x01
    hk1v5c_warn = (status >> 18) & 0x01
    hk1v5c_danger = (status >> 17) & 0x01
    hk1v8_warn = (status >> 16) & 0x01
    hk1v8_danger = (status >> 15) & 0x01
    hk1v8c_warn = (status >> 14) & 0x01
    hk1v8c_danger = (status >> 13) & 0x01
    fob_saturated = (status >> 5) & 0x01
    fib_saturated = (status >> 4) & 0x01
    mode = (status >> 0) & 0x0F
    icu_temp = ((status >> 6) & 0x7F) << 5

    return Packet0(
        hk1v5_warn,
        hk1v5_danger,
        hk1v5c_warn,
        hk1v5c_danger,
        hk1v8_warn,
        hk1v8_danger,
        hk1v8c_warn,
        hk1v8c_danger,
        fob_saturated,
        fib_saturated,
        mode,
        icu_temp,
    )


@dataclass
class Packet1:
    """Dataclass for packet 1."""

    hk2v5_warn: int
    hk2v5_danger: int
    hk2v5c_warn: int
    hk2v5c_danger: int
    hk3v3: int
    hk3v3_current: int
    pri_isvalid: int


def decode_packet1(status: int) -> Packet1:
    """
    Decode packet 1.

    Parameters
    ----------
    status : int
        24-bit integer value.

    Returns
    -------
    list[int]
        List of extracted bytes.

    Notes
    -----
    Bit  5 → pri_isvalid (1-bit value)
    Bits 4-1 → hk2v5_warn, hk2v5_danger, hk2v5c_warn, hk2v5c_danger (4-bit value)
    Bits 16-9 → hk3v3 (8-bit value)
    Bits 8-0 → hk3v3_current (9-bit value)
    """
    hk2v5_warn = (status >> 20) & 0x01
    hk2v5_danger = (status >> 19) & 0x01
    hk2v5c_warn = (status >> 18) & 0x01
    hk2v5c_danger = (status >> 17) & 0x01
    hk3v3 = ((status >> 9) & 0xFF) << 4
    hk3v3_current = ((status >> 0) & 0x1FF) << 3
    pri_isvalid = (status >> 21) & 0x01

    return Packet1(
        hk2v5_warn,
        hk2v5_danger,
        hk2v5c_warn,
        hk2v5c_danger,
        hk3v3,
        hk3v3_current,
        pri_isvalid,
    )


@dataclass
class Packet2:
    """Dataclass for packet 2."""

    hkp8v5_warn: int
    hkp8v5_danger: int
    hkp8v5c_warn: int
    hkp8v5c_danger: int
    hkn8v5: int
    hkn8v5_current: int


def decode_packet2(status: int) -> Packet2:
    """
    Decode packet 2.

    Parameters
    ----------
    status : int
        24-bit integer value.

    Returns
    -------
    list[int]
        List of extracted bytes.

    Notes
    -----
    Bits 23-22 → Packet Number (2-bit value)
    Bits 21-17 → Various warning/danger flags (5-bit value)
    Bits 16-9 → hkn8v5 (8-bit value)
    Bits 8-0 → hkn8v5_current (9-bit value)
    """
    hkp8v5_warn = (status >> 20) & 0x01
    kp8v5_danger = (status >> 19) & 0x01
    hkp8v5c_warn = (status >> 18) & 0x01
    hkp8v5c_danger = (status >> 17) & 0x01
    hkn8v5 = ((status >> 9) & 0xFF) << 4
    hkn8v5_current = ((status >> 0) & 0x1FF) << 3
    return Packet2(
        hkp8v5_warn, kp8v5_danger, hkp8v5c_warn, hkp8v5c_danger, hkn8v5, hkn8v5_current
    )


@dataclass
class Packet3:
    """Dataclass for packet 3."""

    fob_temp: int
    fib_temp: int
    fob_range: int
    fib_range: int
    multbit_errs: int
    sec_isvalid: int


def decode_packet3(status: int) -> Packet3:
    """
    Decode packet 3.

    Parameters
    ----------
    status : int
        24-bit integer value.

    Returns
    -------
    list[int]
        List of extracted bytes.

    Notes
    -----
    Bits 20-13 → fob_temp (8-bit value, shifted left by 4)
    Bits 12-5 → fib_temp (8-bit value, shifted left by 4)
    Bits 4-3 → fob_range (2-bit value)
    Bits 2-1 → fib_range (2-bit value)
    Bit 0 → multbit_errs (1-bit value)
    Bit 5 → sec_isvalid (1-bit value, overlapping with fib_temp extraction)
    """
    fob_temp = ((status >> 13) & 0xFF) << 4
    fib_temp = ((status >> 5) & 0xFF) << 4
    fob_range = (status >> 3) & 0x03
    fib_range = (status >> 1) & 0x03
    multbit_errs = (status >> 0) & 0x01
    sec_isvalid = (status >> 21) & 0x01

    return Packet3(fob_temp, fib_temp, fob_range, fib_range, multbit_errs, sec_isvalid)
