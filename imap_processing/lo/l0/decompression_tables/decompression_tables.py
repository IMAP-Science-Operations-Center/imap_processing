"""Compression tables for decompompressing Lo L0 data."""

from collections import namedtuple

# All TOFs and Checksum values that exist for a case
# must be shifted by 1 bit to the left
DE_BIT_SHIFT = {
    "tof0": 1,
    "tof1": 1,
    "tof2": 1,
    "tof3": 1,
    "cksm": 1,
    "pos": 0,
}

# Named tuple for all the packet fields in the
# direct event packet that need to be parsed
# but are not part of the compressed DE data.
PacketFields = namedtuple(
    "PacketFields",
    [
        "passes",
    ],
)

# Named tuple for all the fixed fields in the
# direct event. These fields will always be transmitted.
FixedFields = namedtuple(
    "FixedFields",
    [
        "coincidence_type",
        "de_time",
        "esa_step",
        "mode",
    ],
)

# Named tuple for all the variable fields in the
# direct event. These fields may or may not be transmitted
# depending on the case and mode.
VariableFields = namedtuple(
    "VariableFields",
    [
        "tof0",
        "tof1",
        "tof2",
        "tof3",
        "cksm",
        "pos",
    ],
)

# number of bits for each packet field
# passes: 32 bits
PACKET_FIELD_BITS = PacketFields(32)

# number of bits for each fixed field
# coincidence_type: 4 bits
# de_time: 12 bits
# esa_step: 3 bits
# mode: 1 bit
FIXED_FIELD_BITS = FixedFields(4, 12, 3, 1)

# number of bits for each variable field if it is transmitted
# tof0: 10 bits
# tof1: 9 bits
# tof2: 9 bits
# tof3: 6 bits
# cksm: 4 bits
# pos: 2 bits
VARIABLE_FIELD_BITS = VariableFields(10, 9, 9, 6, 4, 2)


# Variable fields that are transmitted for each case and mode.
# (case, mode): tof0, tof1, tof2, tof3, cksm, pos
CASE_DECODER = {
    (0, 1): VariableFields(True, False, True, True, True, False),
    (0, 0): VariableFields(True, True, True, True, False, False),
    (1, 0): VariableFields(True, True, True, False, False, False),
    (2, 0): VariableFields(True, True, False, False, False, True),
    (3, 0): VariableFields(True, False, False, False, False, False),
    (4, 1): VariableFields(True, False, True, False, False, True),
    (4, 0): VariableFields(True, False, False, True, False, False),
    (5, 0): VariableFields(True, False, True, False, False, False),
    (6, 1): VariableFields(True, False, False, False, False, True),
    (6, 0): VariableFields(True, False, False, True, False, False),
    (7, 0): VariableFields(True, False, False, False, False, False),
    (8, 0): VariableFields(False, True, True, False, False, True),
    (9, 0): VariableFields(False, True, True, False, False, False),
    (10, 1): VariableFields(False, True, False, False, False, True),
    (10, 0): VariableFields(False, True, False, True, False, False),
    (11, 0): VariableFields(False, True, False, False, False, False),
    (12, 1): VariableFields(False, False, True, False, False, True),
    (12, 0): VariableFields(False, False, True, True, False, False),
    (13, 0): VariableFields(False, False, True, False, False, False),
}
