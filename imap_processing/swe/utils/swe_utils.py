"""Various utility classes and functions to support SWE processing."""

from enum import IntEnum

import pandas as pd

from imap_processing import imap_module_directory

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


class SWEAPID(IntEnum):
    """Create ENUM for apid."""

    SWE_SCIENCE = 1344


def read_lookup_table() -> pd.DataFrame:
    """
    Read lookup table.

    Returns
    -------
    esa_table : pandas.DataFrame
        ESA table.
    """
    # Read lookup table
    lookup_table_path = imap_module_directory / "swe/l1b/swe_esa_lookup_table.csv"
    esa_table = pd.read_csv(lookup_table_path)
    return esa_table
