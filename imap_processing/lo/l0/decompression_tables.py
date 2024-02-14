"""Compression tables for decompompressing Lo L0 data."""
from collections import namedtuple

TOFData = namedtuple(
    "TOFData", ["ENERGY", "POS", "TOF0", "TOF1", "TOF2", "TOF3", "CKSM", "TIME"]
)
tof_decoder_table = {
    # First level of keys are the case numbers
    0: {
        # 1 = Gold Triple
        1: TOFData(
            3,
            0,
            10,
            0,
            9,
            6,
            3,
            12,
        ),
        # 0 = Silver Triple
        0: TOFData(
            3,
            0,
            10,
            9,
            9,
            6,
            0,
            12,
        ),
    },
    1: TOFData(
        3,
        0,
        10,
        9,
        9,
        0,
        0,
        12,
    ),
    2: TOFData(
        3,
        2,
        9,
        9,
        0,
        0,
        0,
        12,
    ),
    3: TOFData(
        3,
        0,
        11,
        0,
        0,
        0,
        0,
        12,
    ),
    4: TOFData(
        3,
        2,
        10,
        0,
        0,
        0,
        0,
        12,
    ),
    5: TOFData(
        3,
        0,
        11,
        0,
        9,
        0,
        0,
        12,
    ),
    6: TOFData(
        3,
        2,
        10,
        0,
        0,
        0,
        0,
        12,
    ),
    7: TOFData(
        3,
        0,
        11,
        0,
        0,
        0,
        0,
        12,
    ),
    8: TOFData(
        3,
        2,
        0,
        9,
        9,
        0,
        0,
        12,
    ),
    9: TOFData(
        3,
        0,
        0,
        10,
        10,
        0,
        0,
        12,
    ),
    10: TOFData(
        3,
        2,
        0,
        10,
        0,
        0,
        0,
        12,
    ),
    11: TOFData(
        3,
        0,
        0,
        11,
        0,
        0,
        0,
        12,
    ),
    12: TOFData(
        3,
        2,
        0,
        0,
        10,
        0,
        0,
        12,
    ),
    13: TOFData(
        3,
        0,
        0,
        0,
        11,
        0,
        0,
        12,
    ),
}

tof_calculation_table = [
    # Case 0
    TOFData("0x0003", "", "0x07FE", "", "0x03FE", "0x007E", "0x00E", "0x0FFF"),
    # Case 1
    TOFData("0x0003", "", "0x07FE", "0x03FE", "0x03FE", "", "", "0x0FFF"),
    # Case 2
    TOFData("0x0FFF", "0x0003", "0x07FC", "0x07FE", "", "", "", "0x0FFF"),
    # Case 3
    TOFData("0x0003", "", "0x07FE", "0x07FE", "", "", "", "0x0FFF"),
    # Case 4
    TOFData("0x0003", "0x0003", "0x07FC", "", "0x03FE", "", "", "0x0FFF"),
    # Case 5
    TOFData("0x0003", "", "0x07FE", "", "0x03FE", "", "", "0x0FFF"),
    # Case 6
    TOFData("0x0003", "0x0003", "0x07FE", "", "", "", "", "0x0FFF"),
    # Case 7
    TOFData("0x0003", "", "0x07FE", "", "", "", "", "0x0FFF"),
    # Case 8
    TOFData("0x0003", "0x0003", "", "0x07FE", "0x03FE", "", "", "0x0FFF"),
    # Case 9
    TOFData("0x0003", "", "", "0x07FE", "0x03FF", "", "", "0x0FFF"),
    # Case 10
    TOFData("0x0003", "0x0003", "", "0x07FE", "", "", "", "0x0FFF"),
    # Case 11
    TOFData("0x0003", "", "", "0x07FE", "", "", "", "0x0FFF"),
    # Case 12
    TOFData("0x0003", "0x0003", "", "", "0x03FF", "", "", "0x0FFF"),
    # Case 13
    TOFData("0x0003", "", "", "", "0x07FE", "", "", "0x0FFF"),
]

tof_coefficient_table = [
    327.68,
    163.84,
    81.82,
    40.96,
    20.48,
    10.24,
    5.12,
    2.56,
    1.28,
    0.64,
    0.32,
    0.16,
]
