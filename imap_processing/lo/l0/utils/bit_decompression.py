"""Decompression for 8 to 12, 8 to 16, and 12 to 16 bits."""

from enum import Enum
from pathlib import Path

import numpy as np


class Decompress(Enum):
    """The decompression options."""

    DECOMPRESS8TO12 = "8_to_12"
    DECOMPRESS8TO16 = "8_to_16"
    DECOMPRESS12TO16 = "12_to_16"


# Load all decompression tables into a dictionary.
DECOMPRESSION_TABLES = {
    enum: np.loadtxt(
        Path(__file__).parent.parent
        / f"decompression_tables/log10_{enum.value}_bit_uncompress.csv",
        delimiter=",",
        skiprows=1,
    )
    for enum in Decompress
}


def decompress_int(
    compressed_values: list, decompression: Decompress, decompression_lookup: dict
) -> list[int]:
    # No idea what the correct type is for the return. Mypy says it is Any
    """
    Will decompress a data field using a specified bit conversion.

    Parameters
    ----------
    compressed_values : list
        Compressed integers.
    decompression : Decompress
        The decompression to use.
    decompression_lookup : dict
        Dictionary containing all the decompression tables.

    Returns
    -------
    decompressed : list[int]
        The decompressed integer.
    """
    valid_decompression = [
        Decompress.DECOMPRESS8TO12,
        Decompress.DECOMPRESS8TO16,
        Decompress.DECOMPRESS12TO16,
    ]
    if decompression not in valid_decompression:
        raise ValueError(
            "Invalid decompression method. Must be one of the following Enums: "
            + "Decompress.DECOMPRESS8TO12, Decompress.DECOMPRESS8TO12, "
            + "Decompress.DECOMPRESS8TO12"
        )
    data = decompression_lookup[decompression]
    # use 2 for Mean column in table
    decompressed: list[int] = data[compressed_values, 2]
    return decompressed
