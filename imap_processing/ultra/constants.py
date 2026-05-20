"""Module for constants and useful shared classes used in Ultra."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from imap_processing import imap_module_directory

SPICE_DATA_SIM_PATH = imap_module_directory / "ultra/l1c/sim_spice_kernels"


@dataclass(frozen=True)
class UltraConstants:
    """
    Constants for Ultra which can be used across different levels or classes.

    Attributes
    ----------
    D_SLIT_FOIL: float
        Shortest distance from slit to foil [mm]
    SLIT_Z: float
        Position of slit on Z axis [mm]
    YF_ESTIMATE_LEFT: float
        Front position of particle for left shutter [mm]
    YF_ESTIMATE_RIGHT: float
        Front position of particle for right shutter [mm]
    N_ELEMENTS: int
        Number of elements in lookup table
    TRIG_CONSTANT: float
        Trigonometric constant [mm]
    COMPOSITE_ENERGY_THRESHOLD: int
        DN threshold for composite energy
    Z_DSTOP: float
        Position of stop foil on Z axis [mm]
    Z_DS: float
        Position of slit on Z axis [mm]
    DF: float
        Distance from slit to foil [mm]
    DMIN: float
        Minimum distance between front and back detectors [mm]
    DMIN_SSD_CTOF: float
        SSD-specific correction to DMIN for time-of-flight normalization
    """

    D_SLIT_FOIL: float = 3.39
    SLIT_Z: float = 44.89
    YF_ESTIMATE_LEFT: float = 40.0
    YF_ESTIMATE_RIGHT: float = -40.0
    N_ELEMENTS: int = 256
    TRIG_CONSTANT: float = 81.92

    # Composite energy threshold for SSD events
    COMPOSITE_ENERGY_THRESHOLD: int = 1707
    # Geometry-related constants
    Z_DSTOP: float = 2.6 / 2  # Position of stop foil on Z axis [mm]
    Z_DS: float = 46.19 - (2.6 / 2)  # Position of slit on Z axis [mm]
    DF: float = 3.39  # Distance from slit to foil [mm]
    # Derived constants
    DMIN_PH_CTOF: float = (
        Z_DS - (2**0.5) * DF
    )  # Minimum distance between front and back detectors [mm]
    DMIN_SSD_CTOF: float = (DMIN_PH_CTOF**2) / (
        DMIN_PH_CTOF - Z_DSTOP
    )  # SSD-specific correction to DMIN [mm]

    # Conversion factors
    KEV_J = 1.602177e-16  # keV to joules
    J_KEV = 1 / KEV_J  # joules to keV
    MASS_H = 1.6735575e-27  # Mass of a hydrogen atom in kilograms.

    # Energy bin constants
    ALPHA = 0.2  # deltaE/E
    ENERGY_START = 3.385  # energy start for the Ultra grids
    N_BINS = 23  # number of energy bins

    # Constants for species determination based on ctof range.
    CTOF_SPECIES_MIN = 50
    CTOF_SPECIES_MAX = 200

    # RPMs for the Ultra instrument.
    # TODO: this is a placeholder.
    CULLING_RPM_MIN = 2.0
    CULLING_RPM_MAX = 6.0

    # Energy Bounds for culling (keV).
    CULLING_ENERGY_BIN_EDGES: ClassVar[list] = [
        3.0,
        10.0,
        20.0,
        50.0,
        300.0,
        1e5,
    ]
    # Counts at l1c are sampled at a finer resolution.
    L1C_COUNTS_NSIDE = 128

    PSET_ENERGY_BIN_EDGES: ClassVar[list] = [
        3.0,
        3.4,
        3.8,
        4.2,
        4.6,
        5.19,
        5.78,
        6.37,
        6.96,
        7.7875,
        8.615,
        9.4425,
        10.27,
        11.63,
        12.99,
        14.35,
        15.71,
        17.3637,
        19.1914,
        21.2116,
        23.4444,
        25.9122,
        28.6398,
        31.6545,
        34.9866,
        38.6694,
        42.7399,
        47.2388,
        52.2113,
        57.7072,
        63.7817,
        70.4955,
        77.9161,
        86.1178,
        95.1828,
        105.202,
        116.276,
        128.516,
        142.044,
        156.995,
        173.521,
        191.787,
        211.975,
        234.288,
        258.95,
        286.208,
        316.335,
    ]

    # Valid event filter constants
    # Note these appear similar to image params constants
    # but they should be used only for the valid event filter.
    ETOFOFF1_EVENTFILTER = 100
    ETOFOFF2_EVENTFILTER = -50
    ETOFSLOPE1_EVENTFILTER = 6667
    ETOFSLOPE2_EVENTFILTER = 7500
    ETOFMAX_EVENTFILTER = 100
    ETOFMIN_EVENTFILTER = 0
    TOFDIFFTPMIN_EVENTFILTER = 226
    TOFDIFFTPMAX_EVENTFILTER = 266

    TOFXE_SPECIES_GROUPS: ClassVar[dict[str, list[int]]] = {
        "proton": [3],
        "non_proton": [20, 28, 36],
    }
    TOFXPH_SPECIES_GROUPS: ClassVar[dict[str, list[int]]] = {
        "proton": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "non_proton": [20, 21, 22, 23, 24, 25, 26],
    }

    SIM_KERNELS_FOR_HELIO_INDEX_MAPS: ClassVar[list] = [
        str(SPICE_DATA_SIM_PATH / k)
        for k in [
            "imap_sclk_0000.tsc",
            "naif0012.tls",
            "imap_spk_demo.bsp",
            "sim_1yr_imap_attitude.bc",
            "imap_001.tf",
            "imap_science_120.tf",
            "sim_1yr_imap_pointing_frame.bc",
        ]
    ]

    FOV_THETA_OFFSET_DEG = 0.0
    FOV_PHI_LIMIT_DEG = 60.0
    # Restricted FOV theta/phi acceptance limits (degrees).
    # Samples outside these bounds are excluded from GF, efficiency, exposure,
    # and counts maps at L1C (fine energy bin maps only).
    RESTRICTED_FOV_THETA_LOW_DEG_45: float = -46.0
    RESTRICTED_FOV_THETA_HIGH_DEG_45: float = 43.0
    RESTRICTED_FOV_THETA_LOW_DEG_90: float = -43.0
    RESTRICTED_FOV_THETA_HIGH_DEG_90: float = 43.0

    # For spatiotemporal culling
    EARTH_RADIUS_KM: float = 6378.1
    N_RE = 60
    DEFAULT_EARTH_CULLING_RADIUS = EARTH_RADIUS_KM * N_RE

    # L1b extended spin culling parameters
    LOW_VOLTAGE_CULL_THRESHOLD = 3400.0
    SPIN_BIN_SIZE = 20
    # Number of energy bins to use in energy dependent culling
    N_CULL_EBINS = 8
    # Bin to start culling at
    BASE_CULL_EBIN = 0
    # Maximum energy threshold in keV. When creating the energy ranges for culling,
    # merge all energy bins above this threshold into one bin.
    MAX_ENERGY_THRESHOLD = 116.0
    # Angle threshold in radians for ULTRA 45 degree culling.
    # This is only needed for ULTRA 45 since Earth may be in the FOV.
    EARTH_ANGLE_45_THRESHOLD = np.radians(15)
    # An array of energy thresholds to use for culling. Each one corresponds to
    # the number of energy bins used.
    # n_bins=len(PSET_ENERGY_BIN_EDGES)[BASE_CULL_EBIN:] // N_CULL_EBINS
    # an error will be raised if this does not match n_bins
    HIGH_ENERGY_CULL_THRESHOLDS = (
        np.array([4.0, 2.0, 1.20, 0.45, 0.1, 0.1]) * SPIN_BIN_SIZE
    )
    # Use the channel defined below to determine which spins are contaminated
    HIGH_ENERGY_CULL_CHANNEL = 5
    # For the high energy cull, we want to combine spin bins because an SEP event is
    # expected to be over a longer time period. Low voltage and statistical culling
    # will still be done on the original spin bins. The variable below defines the
    # radius (in number of spin bins) to use when combining for the high energy cull.
    HIGH_ENERGY_COMBINED_SPIN_BIN_RADIUS = 5
    # Number of iterations to perform for statistical outlier culling.
    STAT_CULLING_N_ITER = 5
    # Sigma threshold to use for statistical outlier culling.
    STAT_CULLING_STD_THRESHOLD = 0.05
    # Energy channels for the upstream ion cull
    # The algorithm will be run twice with the different sets of channels below.
    UPSTREAM_ION_ENERGY_CHANNELS_1: ClassVar[list] = [0, 1, 2]
    UPSTREAM_ION_ENERGY_CHANNELS_2: ClassVar[list] = [2, 3, 4]
    UPSTREAM_SIG_THRESHOLD = 2.5
    # Spectral culling parameters
    SPECTRAL_ENERGY_CHANNELS: ClassVar[list] = [0, 1, 2, 3]
    SPECTRAL_SIG_THRESHOLD = 1
    # Set dimensions for extended spin/goodtime support variables
    # ISTP requires fixed dimensions, so we set these to the maximum we expect to need
    # and pad with fill values if we use fewer bins.
    MAX_ENERGY_RANGES = 16
    MAX_ENERGY_RANGE_EDGES = MAX_ENERGY_RANGES + 1

    # L1C PSET constants

    # When True, applies the FOV restrictions defined above to the L1C fine energy bin
    # maps (GF, efficiency, exposure, counts). This culls regions of the instrument
    # field of view with poor efficiency calibration from inclusion into the map making
    # process.
    APPLY_FOV_RESTRICTIONS_L1C: bool = True

    # When True, applies the boundary scale factors from the ancillary file to exposure
    # time, efficiency, and geometric factor maps.
    APPLY_BOUNDARY_SCALE_FACTORS_L1C: bool = False

    # When True, applies the scattering rejection mask based on the FWHM thresholds
    # to the L1C fine energy bin maps.
    APPLY_SCATTERING_REJECTION_L1C: bool = False
