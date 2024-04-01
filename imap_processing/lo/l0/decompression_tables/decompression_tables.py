"""Compression tables for decompompressing Lo L0 data."""
from collections import namedtuple

# All TOFs and Checksum values that exist for a case
# must be shifted by 1 bit to the left
BIT_SHIFT = 1

DataFields = namedtuple(
    "DataFields",
    [
        "TIME",
        "ENERGY",
        "MODE",
        "TOF0",
        "TOF1",
        "TOF2",
        "TOF3",
        "CKSM",
        "POS",
    ],
)
# the bit length for each field
DATA_BITS = DataFields(12, 3, 1, 10, 9, 9, 6, 3, 1)

TOFFields = namedtuple(
    "TOFFields",
    [
        "TOF0",
        "TOF1",
        "TOF2",
        "TOF3",
        "CKSM",
        "POS",
    ],
)

# TODO:
# For case 0 if mode = 1 TOF1 is not transmitted and is calculated using the checksum
# For case 0 if mode = 0 TOF1 is transmitted
# For cases 4, 6, 10, 12 if mode = 1 TOF3 is not transmitted, but can be calculated
# from the position which is transmitted
# For cases 4, 6, 10, 12 if mode = 0 TOF3 is trasmitted, position is not
# No mode = 1 for case 13? Green highlight?
# No mode = 1 for cases 2, 3, 5, 7, 8, 9, 11

# (case, mode): the TOF fields that are transmitted that case/mode.
CASE_DECODER = {
    (0, 1): TOFFields(True, False, True, True, True, False),
    (0, 0): TOFFields(True, True, True, True, False, False),
    (1, 0): TOFFields(True, True, True, False, False, False),
    (2, 0): TOFFields(True, True, False, False, False, True),
    (3, 0): TOFFields(True, False, False, False, False, False),
    (4, 1): TOFFields(True, False, False, False, False, True),
    (4, 0): TOFFields(True, False, False, True, False, False),
    (5, 0): TOFFields(True, False, True, False, False, False),
    # TODO: 4 and 6 are packed the same?
    (6, 1): TOFFields(True, False, False, True, False, False),
    (6, 0): TOFFields(True, False, False, True, False, False),
    (7, 0): TOFFields(True, False, False, False, False, False),
    (8, 0): TOFFields(False, True, True, False, False, True),
    (9, 0): TOFFields(False, True, True, False, False, False),
    (10, 1): TOFFields(False, True, False, False, False, True),
    (10, 0): TOFFields(False, True, False, True, False, False),
    (11, 0): TOFFields(False, True, False, False, False, False),
    (12, 1): TOFFields(False, False, True, False, False, True),
    (12, 0): TOFFields(False, False, True, True, False, False),
    # TODO: Never 1 for mode?
    (13, 1): TOFFields(),
    (13, 0): TOFFields(False, False, True, False, False, False),
}
