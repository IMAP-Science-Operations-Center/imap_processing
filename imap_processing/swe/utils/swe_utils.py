"""Various utility classes and functions to support SWE processing."""

from enum import IntEnum

import pandas as pd

from imap_processing import imap_module_directory


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
