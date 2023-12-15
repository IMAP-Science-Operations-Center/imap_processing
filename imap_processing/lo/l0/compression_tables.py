# function to get case number
# use case number and data to parse the bits into their respective values
# function to get the hex value and turn the into binary
# get the hex value and determine which values are kept in the second table
# get product sum of the TOF and the kept values in the second table

# decompression:
# compressed data
# uncompressed data
# case number
#
# fine_case_number
# parse_data
# get_hex
#

tof_case_table = {
    "0000": 0,
    "0001": 1,
    "0010": 2,
    "0011": 3,
    "0100": 4,
    "0101": 5,
    "0110": 6,
    "0111": 7,
    "1000": 8,
    "1001": 9,
    "1010": 10,
    "1011": 11,
    "1100": 12,
    "1101": 13,
}

tof_decoder_table = {
    # Case: Energy, Pos, TOF0, TOF1, TOF2, TOF3, CkSm, Time
    0: {
        1: {
            "ENERGY": 3,
            "POS": 0,
            "TOF0": 10,
            "TOF1": 0,
            "TOF2": 9,
            "TOF3": 6,
            "CKSM": 3,
            "TIME": 12,
        },
        0: {
            "ENERGY": 3,
            "POS": 0,
            "TOF0": 10,
            "TOF1": 9,
            "TOF2": 9,
            "TOF3": 6,
            "CKSM": 0,
            "TIME": 12,
        },
    },
    1: {
        "ENERGY": 3,
        "POS": 0,
        "TOF0": 10,
        "TOF1": 9,
        "TOF2": 9,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    2: {
        "ENERGY": 3,
        "POS": 2,
        "TOF0": 9,
        "TOF1": 9,
        "TOF2": 0,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    3: {
        "ENERGY": 3,
        "POS": 0,
        "TOF0": 11,
        "TOF1": 0,
        "TOF2": 0,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    4: {
        "ENERGY": 3,
        "POS": 2,
        "TOF0": 10,
        "TOF1": 0,
        "TOF2": 0,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    5: {
        "ENERGY": 3,
        "POS": 0,
        "TOF0": 11,
        "TOF1": 0,
        "TOF2": 9,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    6: {
        "ENERGY": 3,
        "POS": 2,
        "TOF0": 10,
        "TOF1": 0,
        "TOF2": 0,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    7: {
        "ENERGY": 3,
        "POS": 0,
        "TOF0": 11,
        "TOF1": 0,
        "TOF2": 0,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    8: {
        "ENERGY": 3,
        "POS": 2,
        "TOF0": 0,
        "TOF1": 9,
        "TOF2": 9,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    9: {
        "ENERGY": 3,
        "POS": 0,
        "TOF0": 0,
        "TOF1": 10,
        "TOF2": 10,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    10: {
        "ENERGY": 3,
        "POS": 2,
        "TOF0": 0,
        "TOF1": 10,
        "TOF2": 0,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    11: {
        "ENERGY": 3,
        "POS": 0,
        "TOF0": 0,
        "TOF1": 11,
        "TOF2": 0,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    12: {
        "ENERGY": 3,
        "POS": 2,
        "TOF0": 0,
        "TOF1": 0,
        "TOF2": 10,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
    13: {
        "ENERGY": 3,
        "POS": 0,
        "TOF0": 0,
        "TOF1": 0,
        "TOF2": 11,
        "TOF3": 0,
        "CKSM": 0,
        "TIME": 12,
    },
}

tof_calculation_table = {
    # Case: Time, TOF0, TOF1, TOF2, TOF3, Pos, CkSm
    0: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "0x07FE",
        "TOF1": "",
        "TOF2": "0x03FE",
        "TOF3": "0x007E",
        "POS": "",
        "CKSM": "0x00E",
    },
    1: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "0x07FE",
        "TOF1": "0x03FE",
        "TOF2": "0x03FE",
        "TOF3": "",
        "POS": "",
        "CKSM": "",
    },
    2: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0FFF",
        "TOF0": "0x07FC",
        "TOF1": "0x07FE",
        "TOF2": "",
        "TOF3": "",
        "POS": "0x0003",
        "CKSM": "",
    },
    3: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "0x07FE",
        "TOF1": "0x07FE",
        "TOF2": "",
        "TOF3": "",
        "POS": "",
        "CKSM": "",
    },
    4: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "0x07FC",
        "TOF1": "",
        "TOF2": "0x03FE",
        "TOF3": "",
        "POS": "0x0003",
        "CKSM": "",
    },
    5: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "0x07FE",
        "TOF1": "",
        "TOF2": "0x03FE",
        "TOF3": "",
        "POS": "",
        "CKSM": "",
    },
    6: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "0x07FE",
        "TOF1": "",
        "TOF2": "",
        "TOF3": "",
        "POS": "0x0003",
        "CKSM": "",
    },
    7: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "0x07FE",
        "TOF1": "",
        "TOF2": "",
        "TOF3": "",
        "POS": "",
        "CKSM": "",
    },
    8: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "",
        "TOF1": "0x07FE",
        "TOF2": "0x03FE",
        "TOF3": "",
        "POS": "0x0003",
        "CKSM": "",
    },
    9: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "",
        "TOF1": "0x07FE",
        "TOF2": "0x03FF",
        "TOF3": "",
        "POS": "",
        "CKSM": "",
    },
    10: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "",
        "TOF1": "0x07FE",
        "TOF2": "",
        "TOF3": "",
        "POS": "0x0003",
        "CKSM": "",
    },
    11: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "",
        "TOF1": "0x07FE",
        "TOF2": "",
        "TOF3": "",
        "POS": "",
        "CKSM": "",
    },
    12: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "",
        "TOF1": "",
        "TOF2": "0x03FF",
        "TOF3": "",
        "POS": "0x0003",
        "CKSM": "",
    },
    13: {
        "TIME": "0x0FFF",
        "ENERGY": "0x0003",
        "TOF0": "",
        "TOF1": "",
        "TOF2": "0x07FE",
        "TOF3": "",
        "POS": "",
        "CKSM": "",
    },
}

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


# From IMAP-Lo Compression Tables
bit12_to_bit16 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 23,
    24: 24,
    25: 25,
    26: 26,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 31,
    32: 32,
    33: 33,
    34: 34,
    35: 35,
    36: 36,
    37: 37,
    38: 38,
    39: 39,
    40: 40,
    41: 41,
    42: 42,
    43: 43,
    44: 44,
    45: 45,
    46: 46,
    47: 47,
    48: 48,
    49: 49,
    50: 50,
    51: 51,
    52: 52,
    53: 53,
    54: 54,
    55: 55,
    56: 56,
    57: 57,
    58: 58,
    59: 59,
    60: 60,
    61: 61,
    62: 62,
    63: 63,
    64: 64,
    65: 65,
    66: 66,
    67: 67,
    68: 68,
    69: 69,
    70: 70,
    71: 71,
    72: 72,
    73: 73,
    74: 74,
    75: 75,
    76: 76,
    77: 77,
    78: 78,
    79: 79,
    80: 80,
    81: 81,
    82: 82,
    83: 83,
    84: 84,
    85: 85,
    86: 86,
    87: 87,
    88: 88,
    89: 89,
    90: 90,
    91: 91,
    92: 92,
    93: 93,
    94: 94,
    95: 95,
    96: 96,
    97: 97,
    98: 98,
    99: 99,
    100: 100,
    101: 104,
    102: 109,
    103: 114,
    104: 118,
    105: 123,
    106: 129,
    107: 134,
    108: 140,
    109: 146,
    110: 152,
    111: 159,
    112: 165,
    113: 172,
    114: 180,
    115: 187,
    116: 195,
    117: 204,
    118: 212,
    119: 221,
    120: 231,
    121: 241,
    122: 251,
    123: 262,
    124: 273,
    125: 284,
    126: 296,
    127: 309,
    128: 322,
    129: 336,
    130: 350,
    131: 365,
    132: 381,
    133: 397,
    134: 414,
    135: 432,
    136: 450,
    137: 469,
    138: 489,
    139: 510,
    140: 532,
    141: 555,
    142: 579,
    143: 603,
    144: 629,
    145: 656,
    146: 684,
    147: 713,
    148: 743,
    149: 775,
    150: 808,
    151: 843,
    152: 879,
    153: 916,
    154: 956,
    155: 996,
    156: 1039,
    157: 1083,
    158: 1129,
    159: 1178,
    160: 1228,
    161: 1280,
    162: 1335,
    163: 1392,
    164: 1452,
    165: 1514,
    166: 1578,
    167: 1646,
    168: 1716,
    169: 1789,
    170: 1866,
    171: 1945,
    172: 2029,
    173: 2115,
    174: 2206,
    175: 2300,
    176: 2398,
    177: 2500,
    178: 2607,
    179: 2719,
    180: 2835,
    181: 2956,
    182: 3082,
    183: 3214,
    184: 3351,
    185: 3494,
    186: 3644,
    187: 3799,
    188: 3962,
    189: 4131,
    190: 4307,
    191: 4491,
    192: 4683,
    193: 4883,
    194: 5092,
    195: 5310,
    196: 5536,
    197: 5773,
    198: 6020,
    199: 6277,
    200: 6545,
    201: 6824,
    202: 7116,
    203: 7420,
    204: 7737,
    205: 8068,
    206: 8412,
    207: 8772,
    208: 9147,
    209: 9537,
    210: 9945,
    211: 10370,
    212: 10813,
    213: 11275,
    214: 11756,
    215: 12259,
    216: 12783,
    217: 13329,
    218: 13898,
    219: 14492,
    220: 15111,
    221: 15757,
    222: 16430,
    223: 17132,
    224: 17864,
    225: 18627,
    226: 19423,
    227: 20253,
    228: 21118,
    229: 22021,
    230: 22962,
    231: 23943,
    232: 24966,
    233: 26032,
    234: 27145,
    235: 28304,
    236: 29514,
    237: 30775,
    238: 32090,
    239: 33461,
    240: 34890,
    241: 36381,
    242: 37936,
    243: 39556,
    244: 41247,
    245: 43009,
    246: 44846,
    247: 46763,
    248: 48761,
    249: 50844,
    250: 53016,
    251: 55282,
    252: 57644,
    253: 60107,
    254: 62675,
    255: 65353,
}

# From IMAP-Lo Compression Tables
bit8_to_bit12 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 23,
    24: 24,
    25: 25,
    26: 26,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 31,
    32: 32,
    33: 33,
    34: 34,
    35: 35,
    36: 36,
    37: 37,
    38: 38,
    39: 39,
    40: 40,
    41: 41,
    42: 42,
    43: 43,
    44: 44,
    45: 45,
    46: 46,
    47: 47,
    48: 48,
    49: 49,
    50: 50,
    51: 51,
    52: 52,
    53: 53,
    54: 54,
    55: 55,
    56: 56,
    57: 57,
    58: 58,
    59: 59,
    60: 60,
    61: 61,
    62: 62,
    63: 63,
    64: 64,
    65: 65,
    66: 66,
    67: 67,
    68: 68,
    69: 69,
    70: 70,
    71: 71,
    72: 72,
    73: 73,
    74: 74,
    75: 75,
    76: 76,
    77: 77,
    78: 78,
    79: 79,
    80: 80,
    81: 81,
    82: 82,
    83: 83,
    84: 84,
    85: 85,
    86: 86,
    87: 87,
    88: 88,
    89: 89,
    90: 90,
    91: 91,
    92: 92,
    93: 93,
    94: 94,
    95: 95,
    96: 96,
    97: 97,
    98: 98,
    99: 99,
    100: 100,
    101: 102,
    102: 105,
    103: 107,
    104: 110,
    105: 112,
    106: 115,
    107: 118,
    108: 121,
    109: 124,
    110: 127,
    111: 130,
    112: 133,
    113: 136,
    114: 139,
    115: 143,
    116: 146,
    117: 150,
    118: 153,
    119: 157,
    120: 161,
    121: 165,
    122: 169,
    123: 173,
    124: 177,
    125: 181,
    126: 186,
    127: 190,
    128: 195,
    129: 199,
    130: 204,
    131: 209,
    132: 214,
    133: 219,
    134: 225,
    135: 230,
    136: 236,
    137: 241,
    138: 247,
    139: 253,
    140: 259,
    141: 266,
    142: 272,
    143: 279,
    144: 285,
    145: 292,
    146: 299,
    147: 307,
    148: 314,
    149: 322,
    150: 330,
    151: 337,
    152: 346,
    153: 354,
    154: 363,
    155: 371,
    156: 380,
    157: 390,
    158: 399,
    159: 409,
    160: 419,
    161: 429,
    162: 439,
    163: 450,
    164: 461,
    165: 472,
    166: 483,
    167: 495,
    168: 507,
    169: 519,
    170: 532,
    171: 545,
    172: 558,
    173: 571,
    174: 585,
    175: 599,
    176: 614,
    177: 629,
    178: 644,
    179: 659,
    180: 675,
    181: 692,
    182: 709,
    183: 726,
    184: 743,
    185: 761,
    186: 780,
    187: 799,
    188: 818,
    189: 838,
    190: 858,
    191: 879,
    192: 900,
    193: 922,
    194: 944,
    195: 967,
    196: 991,
    197: 1015,
    198: 1039,
    199: 1064,
    200: 1090,
    201: 1117,
    202: 1144,
    203: 1171,
    204: 1200,
    205: 1229,
    206: 1259,
    207: 1289,
    208: 1320,
    209: 1352,
    210: 1385,
    211: 1419,
    212: 1453,
    213: 1488,
    214: 1524,
    215: 1561,
    216: 1599,
    217: 1638,
    218: 1677,
    219: 1718,
    220: 1760,
    221: 1802,
    222: 1846,
    223: 1891,
    224: 1937,
    225: 1983,
    226: 2032,
    227: 2081,
    228: 2131,
    229: 2183,
    230: 2236,
    231: 2290,
    232: 2345,
    233: 2402,
    234: 2460,
    235: 2520,
    236: 2581,
    237: 2644,
    238: 2708,
    239: 2773,
    240: 2841,
    241: 2910,
    242: 2980,
    243: 3052,
    244: 3126,
    245: 3202,
    246: 3280,
    247: 3359,
    248: 3440,
    249: 3524,
    250: 3609,
    251: 3697,
    252: 3786,
    253: 3878,
    254: 3972,
    255: 4068,
}
