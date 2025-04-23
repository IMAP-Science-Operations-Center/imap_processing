"""HIT L1B constants."""

# Expected number of livestim pulses per integration time.
# This is used to calculate the fractional livetime
LIVESTIM_PULSES = 270

# Fill values for missing data
FILLVAL_FLOAT32 = -1.00e31
FILLVAL_INT64 = -9223372036854775808

# For the L1B summed rates product, counts are summed by particle type,
# energy range, and detector penetration range (Range 2, Range 3, and Range 4).
# See section 6.2 of the algorithm document for more details.

# The counts to sum are in the L2FGRATES, L3FGRATES, and PENFGRATES data
# variables in the L1A product. These variables represent different detector
# ranges for each particle type and energy range.

# Indices at each detector range are provided for each particle type
# and energy range in the dictionary below and the counts at these indices will be
# summed in l1B processing to produce the summed rates product.
# R2 = Indices for Range 2 (L2FGRATES)
# R3 = Indices for Range 3 (L3FGRATES)
# R4 = Indices for Range 4 (PENFGRATES)
# energy_units: MeV/n

SUMMED_PARTICLE_ENERGY_RANGE_MAPPING = {
    "h": [
        {
            "energy_min": 1.8,
            "energy_max": 3.6,
            "R2": [1, 2, 3, 4],
            "R3": [0, 1],
            "R4": [],
        },
        {"energy_min": 4.0, "energy_max": 6.0, "R2": [6, 7], "R3": [3, 4, 5], "R4": []},
        {"energy_min": 6.0, "energy_max": 10.0, "R2": [], "R3": [6, 7], "R4": []},
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [8, 9], "R4": [1]},
    ],
    "he3": [
        {
            "energy_min": 4.0,
            "energy_max": 6.0,
            "R2": [14, 15, 16],
            "R3": [12, 13, 14],
            "R4": [],
        },
        {"energy_min": 6.0, "energy_max": 10.0, "R2": [], "R3": [15, 16], "R4": []},
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [17, 18], "R4": []},
    ],
    "he4": [
        {
            "energy_min": 1.8,
            "energy_max": 3.6,
            "R2": [19, 20, 21, 22],
            "R3": [21],
            "R4": [],
        },
        {
            "energy_min": 4.0,
            "energy_max": 6.0,
            "R2": [24, 25],
            "R3": [23, 24, 25],
            "R4": [],
        },
        {"energy_min": 6.0, "energy_max": 10.0, "R2": [], "R3": [26, 27], "R4": []},
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [28, 29], "R4": [4]},
    ],
    "he": [
        {
            "energy_min": 4.0,
            "energy_max": 6.0,
            "R2": [14, 15, 16, 24, 25],
            "R3": [12, 13, 14, 23, 24, 25],
            "R4": [],
        },
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [],
            "R3": [15, 16, 26, 27],
            "R4": [],
        },
        {
            "energy_min": 10.0,
            "energy_max": 15.0,
            "R2": [],
            "R3": [17, 18, 28, 29],
            "R4": [4],
        },
    ],
    "c": [
        {
            "energy_min": 4.0,
            "energy_max": 6.0,
            "R2": [30, 31, 32],
            "R3": [32, 33],
            "R4": [],
        },
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [33, 34],
            "R3": [34, 35],
            "R4": [],
        },
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [36, 37], "R4": []},
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [38, 39], "R4": [7]},
    ],
    "n": [
        {
            "energy_min": 4.0,
            "energy_max": 6.0,
            "R2": [39, 40, 41],
            "R3": [43],
            "R4": [],
        },
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [42, 43],
            "R3": [44, 45],
            "R4": [],
        },
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [46, 47], "R4": []},
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [48, 49], "R4": [11]},
    ],
    "o": [
        {
            "energy_min": 4.0,
            "energy_max": 6.0,
            "R2": [48, 49, 50],
            "R3": [53],
            "R4": [],
        },
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [51, 52],
            "R3": [54, 55],
            "R4": [],
        },
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [56, 57], "R4": []},
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [58, 59], "R4": []},
    ],
    "ne": [
        {"energy_min": 4.0, "energy_max": 6.0, "R2": [57, 58, 59], "R3": [], "R4": []},
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [60, 61],
            "R3": [63, 64],
            "R4": [],
        },
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [62], "R3": [65, 66], "R4": []},
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [67, 68], "R4": []},
    ],
    "na": [
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [75, 76], "R4": []},
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [77, 78], "R4": []},
    ],
    "mg": [
        {"energy_min": 4.0, "energy_max": 6.0, "R2": [67, 68, 69], "R3": [], "R4": []},
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [70, 71],
            "R3": [83, 84],
            "R4": [],
        },
        {
            "energy_min": 10.0,
            "energy_max": 15.0,
            "R2": [72, 73],
            "R3": [85, 86],
            "R4": [],
        },
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [87, 88], "R4": []},
    ],
    "al": [
        {"energy_min": 6.0, "energy_max": 10.0, "R2": [], "R3": [94, 95], "R4": []},
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [96, 97], "R4": []},
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [98, 99], "R4": []},
    ],
    "si": [
        {"energy_min": 4.0, "energy_max": 6.0, "R2": [78, 79, 80], "R3": [], "R4": []},
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [81, 82],
            "R3": [105, 106],
            "R4": [],
        },
        {
            "energy_min": 10.0,
            "energy_max": 15.0,
            "R2": [83, 84],
            "R3": [107, 108],
            "R4": [],
        },
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [109, 110], "R4": []},
        {
            "energy_min": 27.0,
            "energy_max": 40.0,
            "R2": [],
            "R3": [111, 112],
            "R4": [26],
        },
    ],
    "s": [
        {"energy_min": 4.0, "energy_max": 6.0, "R2": [88, 89, 90], "R3": [], "R4": []},
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [91, 92],
            "R3": [116, 117],
            "R4": [],
        },
        {
            "energy_min": 10.0,
            "energy_max": 15.0,
            "R2": [93, 94],
            "R3": [118, 119],
            "R4": [],
        },
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [120, 121], "R4": []},
        {"energy_min": 27.0, "energy_max": 40.0, "R2": [], "R3": [122, 123], "R4": []},
    ],
    "ar": [
        {"energy_min": 4.0, "energy_max": 6.0, "R2": [98, 99, 100], "R3": [], "R4": []},
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [101, 102],
            "R3": [127, 128],
            "R4": [],
        },
        {
            "energy_min": 10.0,
            "energy_max": 15.0,
            "R2": [103, 104],
            "R3": [129, 130],
            "R4": [],
        },
        {
            "energy_min": 15.0,
            "energy_max": 27.0,
            "R2": [105],
            "R3": [131, 132],
            "R4": [],
        },
        {"energy_min": 27.0, "energy_max": 40.0, "R2": [], "R3": [133, 134], "R4": []},
    ],
    "ca": [
        {
            "energy_min": 4.0,
            "energy_max": 6.0,
            "R2": [109, 110, 111],
            "R3": [],
            "R4": [],
        },
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [112, 113],
            "R3": [138],
            "R4": [],
        },
        {
            "energy_min": 10.0,
            "energy_max": 15.0,
            "R2": [114, 115],
            "R3": [139, 140],
            "R4": [],
        },
        {
            "energy_min": 15.0,
            "energy_max": 27.0,
            "R2": [116],
            "R3": [141, 142],
            "R4": [],
        },
        {"energy_min": 27.0, "energy_max": 40.0, "R2": [], "R3": [143, 144], "R4": []},
    ],
    "fe": [
        {
            "energy_min": 4.0,
            "energy_max": 6.0,
            "R2": [122, 123, 124],
            "R3": [],
            "R4": [],
        },
        {
            "energy_min": 6.0,
            "energy_max": 10.0,
            "R2": [125, 126],
            "R3": [148],
            "R4": [],
        },
        {
            "energy_min": 10.0,
            "energy_max": 15.0,
            "R2": [127, 128],
            "R3": [149, 150],
            "R4": [],
        },
        {
            "energy_min": 15.0,
            "energy_max": 27.0,
            "R2": [129],
            "R3": [151, 152],
            "R4": [],
        },
        {"energy_min": 27.0, "energy_max": 40.0, "R2": [], "R3": [153, 154], "R4": []},
    ],
    "ni": [
        {"energy_min": 10.0, "energy_max": 15.0, "R2": [], "R3": [159, 160], "R4": []},
        {"energy_min": 15.0, "energy_max": 27.0, "R2": [], "R3": [161, 162], "R4": []},
        {"energy_min": 27.0, "energy_max": 40.0, "R2": [], "R3": [163, 164], "R4": []},
    ],
}
