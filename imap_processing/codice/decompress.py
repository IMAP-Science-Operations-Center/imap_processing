"""
Will decompress CoDICE science data.

For CoDICE, there are 3 forms of compression:

    | 1. Table-based lossy compression A (24- or 32-bit -> 8-bit)
    | 2. Table-based lossy compression B (24- or 32-bit -> 8 bit)
    | 3. LZMA lossless compression

Only one lossy option can be selected in cases of lossy + lossless compression.
Thus, there are 6 possibly compression algorithms:

    | 1. No compression
    | 2. Lossy A only
    | 3. Lossy B only
    | 4. Lossless only
    | 5. Lossy A + lossless
    | 6. Lossy B + lossless

In the case of (5) and (6), the data is first run through lossy compression, and
then the result is run through lossless compression. Thus, to decompress, one
must apply lossless decompression first, then lossy decompression

References
----------
    This information was provided via email from Greg Dunn on Oct 23, 2023
"""

import lzma
from enum import IntEnum

from imap_processing.codice.constants import LOSSY_A_TABLE, LOSSY_B_TABLE
from imap_processing.codice.utils import CoDICECompression


def _apply_lossy_a(compressed_bytes: bytes) -> list[int]:
    """
    Apply 8-bit to 32-bit Lossy A decompression algorithm.

    The Lossy A algorithm uses a lookup table imported into this module.

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed byte stream.

    Returns
    -------
    decompressed_values : list[int]
        The 24- or 32-bit decompressed values.
    """
    compressed_values = list(compressed_bytes)
    decompressed_values = [LOSSY_A_TABLE[item] for item in compressed_values]
    return decompressed_values


def _apply_lossy_b(compressed_bytes: bytes) -> list[int]:
    """
    Apply 8-bit to 32-bit Lossy B decompression algorithm.

    The Lossy B algorithm uses a lookup table imported into this module.

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed byte stream.

    Returns
    -------
    decompressed_values : list[int]
        The 24- or 32-bit decompressed values.
    """
    compressed_values = list(compressed_bytes)
    decompressed_values = [LOSSY_B_TABLE[item] for item in compressed_values]
    return decompressed_values


def _apply_lzma_lossless(compressed_bytes: bytes) -> bytes:
    """
    Apply LZMA lossless decompression algorithm.

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed byte stream.

    Returns
    -------
    lzma_decompressed_values : bytes
        The 24- or 32-bit lzma decompressed values.
    """
    lzma_decompressed_values = lzma.decompress(compressed_bytes)

    return lzma_decompressed_values


def decompress(compressed_binary: str, algorithm: IntEnum) -> list[int]:
    """
    Perform decompression on a binary string into a list of integers.

    Apply the appropriate decompression algorithm(s) based on the value
    of the ``algorithm`` attribute. One or more individual algorithms may be
    applied to a given compressed value.

    Parameters
    ----------
    compressed_binary : str
        The compressed binary string.
    algorithm : int
        The algorithm to apply. Supported algorithms are provided in the
        ``codice_utils.CoDICECompression`` class.

    Returns
    -------
    decompressed_values : list[int]
        The 24- or 32-bit decompressed values.
    """
    # Convert the binary string to a byte stream
    compressed_bytes = int(compressed_binary, 2).to_bytes(
        (len(compressed_binary) + 7) // 8, byteorder="big"
    )

    # Apply the appropriate decompression algorithm
    if algorithm == CoDICECompression.NO_COMPRESSION:
        decompressed_values = list(compressed_bytes)
    elif algorithm == CoDICECompression.LOSSY_A:
        decompressed_values = _apply_lossy_a(compressed_bytes)
    elif algorithm == CoDICECompression.LOSSY_B:
        decompressed_values = _apply_lossy_b(compressed_bytes)
    elif algorithm == CoDICECompression.LOSSLESS:
        decompressed_bytes = _apply_lzma_lossless(compressed_bytes)
        decompressed_values = list(decompressed_bytes)
    elif algorithm == CoDICECompression.LOSSY_A_LOSSLESS:
        decompressed_bytes = _apply_lzma_lossless(compressed_bytes)
        decompressed_values = _apply_lossy_a(decompressed_bytes)
    elif algorithm == CoDICECompression.LOSSY_B_LOSSLESS:
        decompressed_bytes = _apply_lzma_lossless(compressed_bytes)
        decompressed_values = _apply_lossy_b(decompressed_bytes)
    else:
        raise ValueError(f"{algorithm} is not supported")

    return decompressed_values
