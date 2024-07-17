"""
Various classes and functions used throughout CoDICE processing.

This module contains utility classes and functions that are used by various
other CoDICE processing modules.
"""

from enum import IntEnum

import space_packet_parser


class CODICEAPID(IntEnum):
    """Create ENUM for CoDICE APIDs."""

    COD_AUT = 1120
    COD_BOOT_HK = 1121
    COD_BOOT_MEMDMP = 1122
    COD_COUNTS_COMMON = 1135
    COD_NHK = 1136
    COD_EVTMSG = 1137
    COD_MEMDMP = 1138
    COD_SHK = 1139
    COD_RTS = 1141
    COD_DIAG_CDHFPGA = 1144
    COD_DIAG_SNSR_HV = 1145
    COD_DIAG_OPTC_HV = 1146
    COD_DIAG_APDFPGA = 1147
    COD_DIAG_SSDFPGA = 1148
    COD_DIAG_FSW = 1149
    COD_DIAG_SYSVARS = 1150
    COD_LO_IAL = 1152
    COD_LO_PHA = 1153
    COD_LO_SW_PRIORITY_COUNTS = 1155
    COD_LO_SW_SPECIES_COUNTS = 1156
    COD_LO_NSW_SPECIES_COUNTS = 1157
    COD_LO_SW_ANGULAR_COUNTS = 1158
    COD_LO_NSW_ANGULAR_COUNTS = 1159
    COD_LO_NSW_PRIORITY_COUNTS = 1160
    COD_LO_INST_COUNTS_AGGREGATED = 1161
    COD_LO_INST_COUNTS_SINGLES = 1162
    COD_HI_IAL = 1168
    COD_HI_PHA = 1169
    COD_HI_INST_COUNTS_AGGREGATED = 1170
    COD_HI_INST_COUNTS_SINGLES = 1171
    COD_HI_OMNI_SPECIES_COUNTS = 1172
    COD_HI_SECT_SPECIES_COUNTS = 1173
    COD_CSTOL_CONFIG = 2457


class CoDICECompression(IntEnum):
    """Create ENUM for CoDICE compression algorithms."""

    NO_COMPRESSION = 0
    LOSSY_A = 1
    LOSSY_B = 2
    LOSSLESS = 3
    LOSSY_A_LOSSLESS = 4
    LOSSY_B_LOSSLESS = 5


def add_metadata_to_array(packet: space_packet_parser, metadata_arrays: dict) -> dict:
    """
    Add metadata to the metadata_arrays.

    Parameters
    ----------
    packet : space_packet_parser.parser.Packet
        CODICE data packet.
    metadata_arrays : dict
        Metadata arrays.

    Returns
    -------
    metadata_arrays : dict
        Updated metadata arrays with values.
    """
    ignore_list = [
        "SPARE_1",
        "SPARE_2",
        "SPARE_3",
        "SPARE_4",
        "SPARE_5",
        "SPARE_6",
        "CHECKSUM",
    ]

    for key, value in packet.header.items():
        metadata_arrays.setdefault(key, []).append(value.raw_value)

    for key, value in packet.data.items():
        if key not in ignore_list:
            metadata_arrays.setdefault(key, []).append(value.raw_value)

    return metadata_arrays
