"""Contains constants variables to support CoDICE processing.

The ``plan_id``, ``plan_step``, and ``view_id`` mentioned in this module are
derived from the packet data.

Acronyms
--------
SW = SunWard
NSW = Non-SunWard
PUI = PickUp Ion
ESA = ElectroStatic Analyzer
"""

from imap_processing.codice.utils import CoDICECompression

# TODO: Convert these to lists because we dont need the fieldname or catdesc
# CDF-friendly names for lo data products
LO_INST_COUNTS_AGGREGATED_NAMES = {
    "aggregated": {"fieldname": "Rates - Aggregated", "catdesc": "Aggregated Rates"}
}

LO_SW_ANGULAR_NAMES = {
    "hplus": {"fieldname": "SW - H+", "catdesc": "Sunward H+ Species"},
    "heplusplus": {"fieldname": "SW - He++", "catdesc": "Sunward He++ Species"},
    "oplus6": {"fieldname": "SW - O+6", "catdesc": "Sunward O+6 Species"},
    "fe_loq": {"fieldname": "SW - Fe lowQ", "catdesc": "Sunward Fe lowQ Species"},
}

LO_NSW_ANGULAR_NAMES = {
    "heplusplus": {"fieldname": "NSW - He++", "catdesc": "Non-sunward He++ Species"},
}

LO_SW_PRIORITY_NAMES = {
    "p0_tcrs": {
        "fieldname": "SW Sector Triple Coincidence PUI's",
        "catdesc": "Sunward Sector Triple Coincidence Pickup Ions Priority",
    },
    "p1_hplus": {"fieldname": "SW Sector H+", "catdesc": "Sunward Sector H+ Priority"},
    "p2_heplusplus": {
        "fieldname": "SW Sector He++",
        "catdesc": "Sunward Sector He++ Priority",
    },
    "p3_heavies": {
        "fieldname": "SW Sector High Charge State Heavies",
        "catdesc": "Sunward Sector High Charge State Heavies Priority =",
    },
    "p4_dcrs": {
        "fieldname": "SW Sector Double Coincidence PUI's",
        "catdesc": "Sunward Sector Double Coincidence Pickup Ions Priority",
    },
}

LO_NSW_PRIORITY_NAMES = {
    "p5_heavies": {
        "fieldname": "NSW Sector Heavies",
        "catdesc": "Non-sunward Sector Heavies Priority",
    },
    "p6_hplus_heplusplus": {
        "fieldname": "NSW H+ and He++",
        "catdesc": "Non-sunward H+ and He++ Priority",
    },
}

LO_SW_SPECIES_NAMES = {
    "hplus": {"fieldname": "SW - H+", "catdesc": "H+ Sunward Species"},
    "heplusplus": {"fieldname": "SW - He+", "catdesc": "He+ Sunward Species"},
    "cplus4": {"fieldname": "SW - C+4", "catdesc": "C+4 Sunward Species"},
    "cplus5": {"fieldname": "SW - C+5", "catdesc": "C+5 Sunward Species"},
    "cplus6": {"fieldname": "SW - C+6", "catdesc": "C+6 Sunward Species"},
    "oplus5": {"fieldname": "SW - O+5", "catdesc": "O+5 Sunward Species"},
    "oplus6": {"fieldname": "SW - O+6", "catdesc": "O+6 Sunward Species"},
    "oplus7": {"fieldname": "SW - O+7", "catdesc": "O+7 Sunward Species"},
    "oplus8": {"fieldname": "SW - O+8", "catdesc": "O+8 Sunward Species"},
    "ne": {"fieldname": "SW - Ne", "catdesc": "Ne Sunward Species"},
    "mg": {"fieldname": "SW - Mg", "catdesc": "Mg Sunward Species"},
    "si": {"fieldname": "SW - Si", "catdesc": "Si Sunward Species"},
    "fe_loq": {"fieldname": "SW - Fe lowQ", "catdesc": "Fe lowQ Sunward Species"},
    "fe_hiq": {"fieldname": "SW - Fe highQ", "catdesc": "Fe highQ Sunward Species"},
    "heplus": {
        "fieldname": "SW - He+ (PUI)",
        "catdesc": "He+ Pickup Ion Sunward Species",
    },
    "cnoplus": {
        "fieldname": "SW - CNO+ (PUI)",
        "catdesc": "CNO+ Pickup Ion Sunward Species",
    },
}

LO_NSW_SPECIES_NAMES = {
    "hplus": {"fieldname": "NSW - H+", "catdesc": "H+ Non-sunward Species"},
    "heplusplus": {"fieldname": "NSW - He++", "catdesc": "He++ Non-sunward Species"},
    "c": {"fieldname": "NSW - C", "catdesc": "C Non-sunward Species"},
    "o": {"fieldname": "NSW - O", "catdesc": "O Non-sunward Species"},
    "ne_si_mg": {
        "fieldname": "NSW - Ne_Si_Mg",
        "catdesc": "Ne-Si-Mg Non-sunward Species",
    },
    "fe": {"fieldname": "NSW - Fe", "catdesc": "Fe Non-sunward Species"},
    "heplus": {"fieldname": "NSW - He+", "catdesc": "He+ Non-sunward Species"},
    "cnoplus": {"fieldname": "NSW - CNO+", "catdesc": "CNO+ Non-sunward Species"},
}

# Compression ID lookup table for Lo data products
# The key is the view_id and the value is the ID for the compression algorithm
# (see utils.CoDICECompression to see how the values correspond)
LO_COMPRESSION_ID_LOOKUP = {
    0: CoDICECompression.LOSSY_A_LOSSLESS,
    1: CoDICECompression.LOSSY_B_LOSSLESS,
    2: CoDICECompression.LOSSY_B_LOSSLESS,
    3: CoDICECompression.LOSSY_A_LOSSLESS,
    4: CoDICECompression.LOSSY_A_LOSSLESS,
    5: CoDICECompression.LOSSY_A_LOSSLESS,
    6: CoDICECompression.LOSSY_A_LOSSLESS,
    7: CoDICECompression.LOSSY_A_LOSSLESS,
    8: CoDICECompression.LOSSY_A_LOSSLESS,
}

# Compression ID lookup table for Hi data products
# The key is the view_id and the value is the ID for the compression algorithm
# (see utils.CoDICECompression to see how the values correspond)
HI_COMPRESSION_ID_LOOKUP = {
    0: CoDICECompression.LOSSY_A,
    1: CoDICECompression.LOSSY_A,
    2: CoDICECompression.LOSSY_A,
    3: CoDICECompression.LOSSY_B_LOSSLESS,
    4: CoDICECompression.LOSSY_B_LOSSLESS,
    5: CoDICECompression.LOSSY_A_LOSSLESS,
    6: CoDICECompression.LOSSY_A_LOSSLESS,
    7: CoDICECompression.LOSSY_A_LOSSLESS,
    8: CoDICECompression.LOSSY_A_LOSSLESS,
    9: CoDICECompression.LOSSY_A_LOSSLESS,
}

# Collapse table ID lookup table for Lo data products
# The key is the view_id and the value is the ID for the collapse table
LO_COLLAPSE_TABLE_ID_LOOKUP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}

# Collapse table ID lookup table for Hi data products
# The key is the view_id and the value is the ID for the collapse table
Hi_COLLAPSE_TABLE_ID_LOOKUP = {
    0: 8,
    1: 9,
    2: 10,
    3: 0,
    4: 1,
    5: 2,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
}

# ESA Sweep table ID lookup table
# The combination of plan_id and plan_step determine the ESA sweep Table to use
# Currently, ESA sweep table 0 is used for every plan_id/plan_step combination,
# but may change in the future. These values are provided in the SCI-LUT excel
# spreadsheet
ESA_SWEEP_TABLE_ID_LOOKUP = {
    (0, 0): 0,
    (0, 1): 0,
    (0, 2): 0,
    (0, 3): 0,
    (1, 0): 0,
    (1, 1): 0,
    (1, 2): 0,
    (1, 3): 0,
    (2, 0): 0,
    (2, 1): 0,
    (2, 2): 0,
    (2, 3): 0,
    (3, 0): 0,
    (3, 1): 0,
    (3, 2): 0,
    (3, 3): 0,
    (4, 0): 0,
    (4, 1): 0,
    (4, 2): 0,
    (4, 3): 0,
    (5, 0): 0,
    (5, 1): 0,
    (5, 2): 0,
    (5, 3): 0,
    (6, 0): 0,
    (6, 1): 0,
    (6, 2): 0,
    (6, 3): 0,
    (7, 0): 0,
    (7, 1): 0,
    (7, 2): 0,
    (7, 3): 0,
}

# Lo Stepping table ID lookup table
# The combination of plan_id and plan_step determine the Lo Stepping Table to
# use. Currently, LO Stepping table 0 is used for every plan_id/plan_step
# combination, but may change in the future. These values are provided in the
# SCI-LUT excel spreadsheet
LO_STEPPING_TABLE_ID_LOOKUP = {
    (0, 0): 0,
    (0, 1): 0,
    (0, 2): 0,
    (0, 3): 0,
    (1, 0): 0,
    (1, 1): 0,
    (1, 2): 0,
    (1, 3): 0,
    (2, 0): 0,
    (2, 1): 0,
    (2, 2): 0,
    (2, 3): 0,
    (3, 0): 0,
    (3, 1): 0,
    (3, 2): 0,
    (3, 3): 0,
    (4, 0): 0,
    (4, 1): 0,
    (4, 2): 0,
    (4, 3): 0,
    (5, 0): 0,
    (5, 1): 0,
    (5, 2): 0,
    (5, 3): 0,
    (6, 0): 0,
    (6, 1): 0,
    (6, 2): 0,
    (6, 3): 0,
    (7, 0): 0,
    (7, 1): 0,
    (7, 2): 0,
    (7, 3): 0,
}

# Lookup tables for Lossy decompression algorithms "A" and "B"
# These were provided by Greg Dunn via his sohis_cdh_utils.v script and then
# transformed into python dictionaries. The values in these tables are subject
# to change, but the format is expected to stay the same.
LOSSY_A_TABLE = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    27: 28,
    28: 29,
    29: 30,
    30: 31,
    31: 32,
    32: 34,
    33: 36,
    34: 38,
    35: 40,
    36: 42,
    37: 44,
    38: 46,
    39: 48,
    40: 50,
    41: 52,
    42: 54,
    43: 56,
    44: 58,
    45: 60,
    46: 62,
    47: 64,
    48: 68,
    49: 72,
    50: 76,
    51: 80,
    52: 84,
    53: 88,
    54: 92,
    55: 96,
    56: 100,
    57: 104,
    58: 108,
    59: 112,
    60: 116,
    61: 120,
    62: 124,
    63: 128,
    64: 136,
    65: 144,
    66: 152,
    67: 160,
    68: 168,
    69: 176,
    70: 184,
    71: 192,
    72: 200,
    73: 208,
    74: 216,
    75: 224,
    76: 232,
    77: 240,
    78: 248,
    79: 256,
    80: 272,
    81: 288,
    82: 304,
    83: 320,
    84: 336,
    85: 352,
    86: 368,
    87: 384,
    88: 400,
    89: 416,
    90: 432,
    91: 448,
    92: 464,
    93: 480,
    94: 496,
    95: 512,
    96: 544,
    97: 576,
    98: 608,
    99: 640,
    100: 672,
    101: 704,
    102: 736,
    103: 768,
    104: 800,
    105: 832,
    106: 864,
    107: 896,
    108: 928,
    109: 960,
    110: 992,
    111: 1024,
    112: 1088,
    113: 1152,
    114: 1216,
    115: 1280,
    116: 1344,
    117: 1408,
    118: 1472,
    119: 1536,
    120: 1600,
    121: 1664,
    122: 1728,
    123: 1792,
    124: 1856,
    125: 1920,
    126: 1984,
    127: 2048,
    128: 2176,
    129: 2304,
    130: 2432,
    131: 2560,
    132: 2688,
    133: 2816,
    134: 2944,
    135: 3072,
    136: 3200,
    137: 3328,
    138: 3456,
    139: 3584,
    140: 3712,
    141: 3840,
    142: 3968,
    143: 4096,
    144: 4352,
    145: 4608,
    146: 4864,
    147: 5120,
    148: 5376,
    149: 5632,
    150: 5888,
    151: 6144,
    152: 6400,
    153: 6656,
    154: 6912,
    155: 7168,
    156: 7424,
    157: 7680,
    158: 7936,
    159: 8192,
    160: 8704,
    161: 9216,
    162: 9728,
    163: 10240,
    164: 10752,
    165: 11264,
    166: 11776,
    167: 12288,
    168: 12800,
    169: 13312,
    170: 13824,
    171: 14336,
    172: 14848,
    173: 15360,
    174: 15872,
    175: 16384,
    176: 17408,
    177: 18432,
    178: 19456,
    179: 20480,
    180: 21504,
    181: 22528,
    182: 23552,
    183: 24576,
    184: 25600,
    185: 26624,
    186: 27648,
    187: 28672,
    188: 29696,
    189: 30720,
    190: 31744,
    191: 32768,
    192: 34816,
    193: 36864,
    194: 38912,
    195: 40960,
    196: 43008,
    197: 45056,
    198: 47104,
    199: 49152,
    200: 51200,
    201: 53248,
    202: 55296,
    203: 57344,
    204: 59392,
    205: 61440,
    206: 63488,
    207: 65536,
    208: 69632,
    209: 73728,
    210: 77824,
    211: 81920,
    212: 86016,
    213: 90112,
    214: 94208,
    215: 98304,
    216: 102400,
    217: 106496,
    218: 110592,
    219: 114688,
    220: 118784,
    221: 122880,
    222: 126976,
    223: 131072,
    224: 139264,
    225: 147456,
    226: 155648,
    227: 163840,
    228: 172032,
    229: 180224,
    230: 188416,
    231: 196608,
    232: 204800,
    233: 212992,
    234: 221184,
    235: 229376,
    236: 237568,
    237: 245760,
    238: 253952,
    239: 262144,
    240: 278528,
    241: 294912,
    242: 311296,
    243: 327680,
    244: 344064,
    245: 360448,
    246: 376832,
    247: 393216,
    248: 409600,
    249: 425984,
    250: 442368,
    251: 458752,
    252: 475136,
    253: 491520,
    254: 507904,
}

LOSSY_B_TABLE = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    27: 28,
    28: 29,
    29: 30,
    30: 31,
    31: 32,
    32: 34,
    33: 36,
    34: 38,
    35: 40,
    36: 42,
    37: 44,
    38: 46,
    39: 48,
    40: 50,
    41: 52,
    42: 54,
    43: 56,
    44: 58,
    45: 60,
    46: 62,
    47: 64,
    48: 68,
    49: 72,
    50: 76,
    51: 80,
    52: 84,
    53: 88,
    54: 92,
    55: 96,
    56: 100,
    57: 104,
    58: 108,
    59: 112,
    60: 116,
    61: 120,
    62: 124,
    63: 128,
    64: 136,
    65: 144,
    66: 152,
    67: 160,
    68: 168,
    69: 176,
    70: 184,
    71: 192,
    72: 200,
    73: 208,
    74: 216,
    75: 224,
    76: 232,
    77: 240,
    78: 248,
    79: 256,
    80: 272,
    81: 288,
    82: 304,
    83: 320,
    84: 336,
    85: 352,
    86: 368,
    87: 384,
    88: 400,
    89: 416,
    90: 432,
    91: 448,
    92: 464,
    93: 480,
    94: 496,
    95: 512,
    96: 544,
    97: 576,
    98: 608,
    99: 640,
    100: 672,
    101: 704,
    102: 736,
    103: 768,
    104: 800,
    105: 832,
    106: 864,
    107: 896,
    108: 928,
    109: 960,
    110: 992,
    111: 1024,
    112: 1088,
    113: 1152,
    114: 1216,
    115: 1280,
    116: 1344,
    117: 1408,
    118: 1472,
    119: 1536,
    120: 1600,
    121: 1664,
    122: 1728,
    123: 1792,
    124: 1856,
    125: 1920,
    126: 1984,
    127: 2048,
    128: 2176,
    129: 2304,
    130: 2432,
    131: 2560,
    132: 2688,
    133: 2816,
    134: 2944,
    135: 3072,
    136: 3200,
    137: 3328,
    138: 3456,
    139: 3584,
    140: 3712,
    141: 3840,
    142: 3968,
    143: 4096,
    144: 4352,
    145: 4608,
    146: 4864,
    147: 5120,
    148: 5376,
    149: 5632,
    150: 5888,
    151: 6144,
    152: 6400,
    153: 6656,
    154: 6912,
    155: 7168,
    156: 7424,
    157: 7680,
    158: 7936,
    159: 8192,
    160: 8704,
    161: 9216,
    162: 9728,
    163: 10240,
    164: 10752,
    165: 11264,
    166: 11776,
    167: 12288,
    168: 12800,
    169: 13312,
    170: 13824,
    171: 14336,
    172: 14848,
    173: 15360,
    174: 15872,
    175: 16384,
    176: 17408,
    177: 18432,
    178: 19456,
    179: 20480,
    180: 21504,
    181: 22528,
    182: 23552,
    183: 24576,
    184: 25600,
    185: 26624,
    186: 27648,
    187: 28672,
    188: 29696,
    189: 30720,
    190: 31744,
    191: 32768,
    192: 36864,
    193: 40960,
    194: 45056,
    195: 49152,
    196: 53248,
    197: 57344,
    198: 61440,
    199: 65536,
    200: 73728,
    201: 81920,
    202: 90112,
    203: 98304,
    204: 106496,
    205: 114688,
    206: 122880,
    207: 131072,
    208: 147456,
    209: 163840,
    210: 180224,
    211: 196608,
    212: 212992,
    213: 229376,
    214: 245760,
    215: 262144,
    216: 294912,
    217: 327680,
    218: 360448,
    219: 393216,
    220: 425984,
    221: 458752,
    222: 491520,
    223: 524288,
    224: 589824,
    225: 655360,
    226: 720896,
    227: 786432,
    228: 851968,
    229: 917504,
    230: 983040,
    231: 1048576,
    232: 1179648,
    233: 1310720,
    234: 1441792,
    235: 1572864,
    236: 1703936,
    237: 1835008,
    238: 1966080,
    239: 2097152,
    240: 2359296,
    241: 2621440,
    242: 2883584,
    243: 3145728,
    244: 3407872,
    245: 3670016,
    246: 3932160,
    247: 4194304,
    248: 4718592,
    249: 5242880,
    250: 5767168,
    251: 6291456,
    252: 6815744,
    253: 7340032,
    254: 7864320,
}
