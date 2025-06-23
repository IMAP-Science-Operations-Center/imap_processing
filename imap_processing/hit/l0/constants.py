"""HIT L0 constants for data decommutation."""

from collections import namedtuple

import numpy as np

# energy_units: MeV/n
MOD_10_MAPPING = {
    0: {"species": "h", "energy_min": 1.8, "energy_max": 3.6},
    1: {"species": "h", "energy_min": 4, "energy_max": 6},
    2: {"species": "h", "energy_min": 6, "energy_max": 10},
    3: {"species": "he4", "energy_min": 4, "energy_max": 6},
    4: {"species": "he4", "energy_min": 6, "energy_max": 12},
    5: {"species": "cno", "energy_min": 4, "energy_max": 6},
    6: {"species": "cno", "energy_min": 6, "energy_max": 12},
    7: {"species": "nemgsi", "energy_min": 4, "energy_max": 6},
    8: {"species": "nemgsi", "energy_min": 6, "energy_max": 12},
    9: {"species": "fe", "energy_min": 4, "energy_max": 12},
}

# Structure to hold binary details for a
# section of science data. Used to unpack
# binary data.
HITPacking = namedtuple(
    "HITPacking",
    [
        "bit_length",
        "section_length",
        "shape",
    ],
)

# Define data structure for counts rates data
COUNTS_DATA_STRUCTURE = {
    # field: bit_length, section_length, shape
    # ------------------------------------------
    # science frame header
    "hdr_unit_num": HITPacking(2, 2, (1,)),
    "hdr_frame_version": HITPacking(6, 6, (1,)),
    "hdr_dynamic_threshold_state": HITPacking(2, 2, (1,)),
    "hdr_leak_conv": HITPacking(1, 1, (1,)),
    "hdr_heater_duty_cycle": HITPacking(4, 4, (1,)),
    "hdr_code_ok": HITPacking(1, 1, (1,)),
    "hdr_minute_cnt": HITPacking(8, 8, (1,)),
    # ------------------------------------------
    # spare bits. Contains no data
    "spare": HITPacking(24, 24, (1,)),
    # ------------------------------------------
    # erates - contains livetime counters
    "livetime_counter": HITPacking(16, 16, (1,)),  # livetime counter
    "num_trig": HITPacking(16, 16, (1,)),  # number of triggers
    "num_reject": HITPacking(16, 16, (1,)),  # number of rejected events
    "num_acc_w_pha": HITPacking(
        16, 16, (1,)
    ),  # number of accepted events with PHA data
    "num_acc_no_pha": HITPacking(16, 16, (1,)),  # number of events without PHA data
    "num_haz_trig": HITPacking(16, 16, (1,)),  # number of triggers with hazard flag
    "num_haz_reject": HITPacking(
        16, 16, (1,)
    ),  # number of rejected events with hazard flag
    "num_haz_acc_w_pha": HITPacking(
        16, 16, (1,)
    ),  # number of accepted hazard events with PHA data
    "num_haz_acc_no_pha": HITPacking(
        16, 16, (1,)
    ),  # number of hazard events without PHA data
    # -------------------------------------------
    "sngrates": HITPacking(16, 1856, (2, 58)),  # single rates
    # -------------------------------------------
    # evprates - contains event processing rates
    "nread": HITPacking(16, 16, (1,)),  # events read from event fifo
    "nhazard": HITPacking(16, 16, (1,)),  # events tagged with hazard flag
    "nadcstim": HITPacking(16, 16, (1,)),  # adc-stim events
    "nodd": HITPacking(16, 16, (1,)),  # odd events
    "noddfix": HITPacking(16, 16, (1,)),  # odd events that were fixed in sw
    "nmulti": HITPacking(
        16, 16, (1,)
    ),  # events with multiple hits in a single detector
    "nmultifix": HITPacking(16, 16, (1,)),  # multi events that were fixed in sw
    "nbadtraj": HITPacking(16, 16, (1,)),  # bad trajectory
    "nl2": HITPacking(16, 16, (1,)),  # events sorted into L12 event category
    "nl3": HITPacking(16, 16, (1,)),  # events sorted into L123 event category
    "nl4": HITPacking(16, 16, (1,)),  # events sorted into L1423 event category
    "npen": HITPacking(16, 16, (1,)),  # events sorted into penetrating event category
    "nformat": HITPacking(16, 16, (1,)),  # nothing currently goes in this slot
    "naside": HITPacking(16, 16, (1,)),  # A-side events
    "nbside": HITPacking(16, 16, (1,)),  # B-side events
    "nerror": HITPacking(16, 16, (1,)),  # events that caused a processing error
    "nbadtags": HITPacking(
        16, 16, (1,)
    ),  # events with inconsistent tags vs pulse heights
    # -------------------------------------------
    # other count rates
    "coinrates": HITPacking(16, 416, (26,)),  # coincidence rates
    "pbufrates": HITPacking(16, 512, (32,)),  # priority buffer rates
    "l2fgrates": HITPacking(16, 2112, (132,)),  # range 2 foreground rates
    "l2bgrates": HITPacking(16, 192, (12,)),  # range 2 background rates
    "l3fgrates": HITPacking(16, 2672, (167,)),  # range 3 foreground rates
    "l3bgrates": HITPacking(16, 192, (12,)),  # range 3 background rates
    "penfgrates": HITPacking(16, 528, (33,)),  # range 4 foreground rates
    "penbgrates": HITPacking(16, 240, (15,)),  # range 4 background rates
    "ialirtrates": HITPacking(16, 320, (20,)),  # ialirt rates
    "sectorates": HITPacking(
        16, 1920, (8, 15)
    ),  # sectored rates (8 zenith angles, 15 azimuth angles)
    "l4fgrates": HITPacking(16, 768, (48,)),  # all range foreground rates
    "l4bgrates": HITPacking(16, 384, (24,)),  # all range foreground rates
}


# Define the pattern of grouping flags in a complete science frame.
FLAG_PATTERN = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2])

# Define size of science frame (num of packets)
FRAME_SIZE = len(FLAG_PATTERN)

# Define the number of bits in the mantissa and exponent for
# decompressing data
MANTISSA_BITS = 12
EXPONENT_BITS = 4

# Define sectorate angles
ZENITH_ANGLES = np.array(
    [11.25, 33.75, 56.25, 78.75, 101.25, 123.75, 146.25, 168.75], dtype=np.float32
)
AZIMUTH_ANGLES = np.array(
    [12, 36, 60, 84, 108, 132, 156, 180, 204, 228, 252, 276, 300, 324, 348],
    dtype=np.float32,
)
