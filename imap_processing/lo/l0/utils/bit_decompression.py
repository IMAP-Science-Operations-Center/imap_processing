import numpy as np
from pathlib import Path
from enum import Enum

class Decompress(Enum):
    """Decompression options."""

    DECOMPRESS8TO12 = "8_to_12"
    DECOMPRESS8TO16 = "8_to_16"
    DECOMPRESS12TO16 = "12_to_16"

def decompress_int(compressed_value, decompression):
    """
    Decompress a data field using a specified bit conversion.

    Parameters
    ----------
    compressed_value : int
        A compressed integer.
    decompression: Decompress
        The decompression to use.

    Returns
    -------
    decompressed : int
        The decompressed integer.
    """
    valid_decompression = [Decompress.DECOMPRESS8TO12, Decompress.DECOMPRESS8TO16, Decompress.DECOMPRESS12TO16]
    if decompression not in valid_decompression:
        raise ValueError("Invalid decompression method. Must be one of the following Enums: " +
                         "Decompress.DECOMPRESS8TO12, Decompress.DECOMPRESS8TO12, Decompress.DECOMPRESS8TO12")

    # load the decompression table for the method specified
    data = np.loadtxt(Path(__file__).parent / f'../decompression_tables/{decompression.value}_bit.csv', delimiter=',', skiprows=1)
    return int(data[compressed_value][1])
