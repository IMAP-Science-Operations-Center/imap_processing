"""Dataclasses for Level 0 MAG I-ALiRT data."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Packet0:
    HK1V5_WARN: int
    HK1V5_DANGER: int
    HK1V5C_WARN: int
    HK1V5C_DANGER: int
    HK1V8_WARN: int
    HK1V8_DANGER: int
    HK1V8C_WARN: int
    HK1V8C_DANGER: int
    FOB_SATURATED: int
    FIB_SATURATED: int
    MODE: int
    ICU_TEMP: int


def decode_packet0(status: int) -> Packet0:
    # Bits 23-22 → Packet Number (2-bit value)
    # Bits 21-17 → HK1V5_WARN, HK1V5_DANGER, HK1V5C_WARN, HK1V5C_DANGER, HK1V8_WARN (5-bit value)
    # Bits 16-13 → HK1V8_DANGER, HK1V8C_WARN, HK1V8C_DANGER (4-bit value)
    # Bit  5 → FOB_SATURATED (1-bit value)
    # Bit  4 → FIB_SATURATED (1-bit value)
    # Bits 3-0 → MODE (4-bit value)
    # Bits 12-6 → ICU_TEMP (7-bit value)
    HK1V5_WARN = (status >> 20) & 0x01
    HK1V5_DANGER = (status >> 19) & 0x01
    HK1V5C_WARN = (status >> 18) & 0x01
    HK1V5C_DANGER = (status >> 17) & 0x01
    HK1V8_WARN = (status >> 16) & 0x01
    HK1V8_DANGER = (status >> 15) & 0x01
    HK1V8C_WARN = (status >> 14) & 0x01
    HK1V8C_DANGER = (status >> 13) & 0x01
    FOB_SATURATED = (status >> 5) & 0x01
    FIB_SATURATED = (status >> 4) & 0x01
    MODE = (status >> 0) & 0x0F
    ICU_TEMP = ((status >> 6) & 0x7F) << 5

    return Packet0(
        HK1V5_WARN, HK1V5_DANGER, HK1V5C_WARN, HK1V5C_DANGER,
        HK1V8_WARN, HK1V8_DANGER, HK1V8C_WARN, HK1V8C_DANGER,
        FOB_SATURATED, FIB_SATURATED, MODE, ICU_TEMP
    )


@dataclass
class Packet1:
    HK2V5_WARN: int
    HK2V5_DANGER: int
    HK2V5C_WARN: int
    HK2V5C_DANGER: int
    HK3V3: int
    HK3V3_CURRENT: int
    pri_isValid: int


def decode_packet1(status: int) -> Packet1:
    # Bit  5 → pri_isValid (1-bit value)
    # Bits 4-1 → HK2V5_WARN, HK2V5_DANGER, HK2V5C_WARN, HK2V5C_DANGER (4-bit value)
    # Bits 16-9 → HK3V3 (8-bit value)
    # Bits 8-0 → HK3V3_CURRENT (9-bit value)
    HK2V5_WARN = (status >> 20) & 0x01
    HK2V5_DANGER = (status >> 19) & 0x01
    HK2V5C_WARN = (status >> 18) & 0x01
    HK2V5C_DANGER = (status >> 17) & 0x01
    HK3V3 = ((status >> 9) & 0xFF) << 4
    HK3V3_CURRENT = ((status >> 0) & 0x1FF) << 3
    pri_isValid = (status >> 5) & 0x01
    return Packet1(HK2V5_WARN, HK2V5_DANGER, HK2V5C_WARN, HK2V5C_DANGER,
                   HK3V3, HK3V3_CURRENT, pri_isValid)


@dataclass
class Packet2:
    HKP8V5_WARN: int
    KP8V5_DANGER: int
    HKP8V5C_WARN: int
    HKP8V5C_DANGER: int
    HKN8V5: int
    HKN8V5_CURRENT: int


def decode_packet2(status: int) -> Packet2:
    # Bits 23-22 → Packet Number (2-bit value)
    # Bits 21-17 → Various warning/danger flags (5-bit value)
    # Bits 16-9 → HKN8V5 (8-bit value)
    # Bits 8-0 → HKN8V5_CURRENT (9-bit value)
    HKP8V5_WARN = (status >> 20) & 0x01
    KP8V5_DANGER = (status >> 19) & 0x01
    HKP8V5C_WARN = (status >> 18) & 0x01
    HKP8V5C_DANGER = (status >> 17) & 0x01
    HKN8V5 = ((status >> 9) & 0xFF) << 4
    HKN8V5_CURRENT = ((status >> 0) & 0x1FF) << 3
    return Packet2(HKP8V5_WARN, KP8V5_DANGER,
                   HKP8V5C_WARN, HKP8V5C_DANGER,
                   HKN8V5, HKN8V5_CURRENT)


@dataclass
class Packet3:
    FOB_TEMP: int
    FIB_TEMP: int
    FOB_RANGE: int
    FIB_RANGE: int
    MULTBIT_ERRS: int
    sec_isValid: int


def decode_packet3(status: int) -> Packet3:
    # Bits 20-13 → FOB_TEMP (8-bit value, shifted left by 4)
    # Bits 12-5 → FIB_TEMP (8-bit value, shifted left by 4)
    # Bits 4-3 → FOB_RANGE (2-bit value)
    # Bits 2-1 → FIB_RANGE (2-bit value)
    # Bit 0 → MULTBIT_ERRS (1-bit value)
    # Bit 5 → sec_isValid (1-bit value, overlapping with FIB_TEMP extraction)
    FOB_TEMP = ((status >> 13) & 0xFF) << 4
    FIB_TEMP = ((status >> 5) & 0xFF) << 4
    FOB_RANGE = (status >> 3) & 0x03
    FIB_RANGE = (status >> 1) & 0x03
    MULTBIT_ERRS = (status >> 0) & 0x01
    sec_isValid = (status >> 5) & 0x01

    return Packet3(FOB_TEMP, FIB_TEMP, FOB_RANGE,
                   FIB_RANGE, MULTBIT_ERRS, sec_isValid)
