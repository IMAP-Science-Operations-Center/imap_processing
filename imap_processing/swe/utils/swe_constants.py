"""Module containing constants used in the SWE processing pipeline."""

import numpy as np

N_ESA_STEPS = 24
N_ANGLE_SECTORS = 30
N_CEMS = 7
N_QUARTER_CYCLES = 4
N_ANGLE_BINS = 30
N_QUARTER_CYCLE_STEPS = 180

MICROSECONDS_IN_SECOND = 1e6

# TODO: add these to instrument status summary
ENERGY_CONVERSION_FACTOR = 4.75
# 7 CEMs geometric factors in cm^2 sr eV/eV units.
GEOMETRIC_FACTORS = np.array(
    [
        424.4e-6,
        564.5e-6,
        763.8e-6,
        916.9e-6,
        792.0e-6,
        667.7e-6,
        425.2e-6,
    ]
)

ELECTRON_MASS = 9.10938356e-31  # kg

# See doc string of calculate_phase_space_density() for more details.
VELOCITY_CONVERSION_FACTOR = 1.237e31
# See doc string of calculate_flux() for more details.
FLUX_CONVERSION_FACTOR = 6.187e30

CEM_DETECTORS_ANGLE = np.array([-63, -42, -21, 0, 21, 42, 63])

# ESA voltage and index in the final data table
ESA_VOLTAGE_ROW_INDEX_DICT = {
    0.56: 0,
    0.78: 1,
    1.08: 2,
    1.51: 3,
    2.10: 4,
    2.92: 5,
    4.06: 6,
    5.64: 7,
    7.85: 8,
    10.92: 9,
    15.19: 10,
    21.13: 11,
    29.39: 12,
    40.88: 13,
    56.87: 14,
    79.10: 15,
    110.03: 16,
    153.05: 17,
    212.89: 18,
    296.14: 19,
    411.93: 20,
    572.99: 21,
    797.03: 22,
    1108.66: 23,
}
