"""
Various classes and functions used throughout SWAPI processing.

This module contains utility classes and functions that are used by various
other SWAPI processing modules.
"""

from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd


class SWAPIAPID(IntEnum):
    """Create ENUM for apid."""

    SWP_HK = 1184
    SWP_SCI = 1188
    SWP_AUT = 1192


class SWAPIMODE(IntEnum):
    """Create ENUM for MODE."""

    LVENG = 0
    LVSCI = 1
    HVENG = 2
    HVSCI = 3


def read_swapi_lut_table(file_path: Path) -> pd.DataFrame:
    """
    Read the LUT table from a CSV file.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the LUT table CSV file.

    Returns
    -------
    pandas.DataFrame
        The LUT table as a DataFrame.
    """
    df = pd.read_csv(file_path)

    # Clean and convert 'Energy' column from comma-separated strings to integers
    df["Energy"] = (
        df["Energy"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("Solve", -1)
        .astype(np.int64)
    )

    return df
