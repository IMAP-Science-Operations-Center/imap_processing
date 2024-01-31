"""Decompression for 8 to 12, 8 to 16, and 12 to 16 bits."""
from enum import Enum
from pathlib import Path

import numpy as np


class Decompress(Enum):
    """Decompression options."""

    DECOMPRESS8TO12 = "8_to_12"
    DECOMPRESS8TO16 = "8_to_16"
    DECOMPRESS12TO16 = "12_to_16"


def decompression_tables():
    """Load all decompression tables into a dictionary."""
    # load all decompression tables
    return {
        enum: np.loadtxt(
            Path(__file__).parent.parent / f"decompression_tables/{enum.value}_bit.csv",
            delimiter=",",
            skiprows=1,
        )
        for enum in Decompress
    }


def decompress_int(compressed_value, decompression, decompression_lookup):
    """
    Decompress a data field using a specified bit conversion.

    Parameters
    ----------
    compressed_value : int
        A compressed integer.
    decompression : Decompress
        The decompression to use.
    decompression_lookup : dict
        dictionary containing all the decompression tables

    Returns
    -------
    decompressed : int
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
    return int(data[compressed_value][1])
