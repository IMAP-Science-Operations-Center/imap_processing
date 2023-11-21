from enum import IntEnum


class CoDICECompression(IntEnum):
    """Create ENUM for CoDICE compression algorithms.

    Parameters
    ----------
    IntEnum : IntEnum
    """

    NO_COMPRESSION = 1
    LOSSY_A = 2
    LOSSY_B = 3
    LOSSLESS = 4
    LOSSY_A_LOSSLESS = 5
    LOSSY_B_LOSSLESS = 6
