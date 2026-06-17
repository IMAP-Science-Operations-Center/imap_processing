"""Ancillary file reading for IMAP-Lo processing."""

from pathlib import Path
from typing import Any

import pandas as pd

# convert the YYYYDDD datetime format directly upon reading
_CONVERTERS = {
    "YYYYDDD": lambda x: pd.to_datetime(str(x), format="%Y%j"),
    "#YYYYDDD": lambda x: pd.to_datetime(str(x), format="%Y%j"),
    "YYYYDDD_strt": lambda x: pd.to_datetime(str(x), format="%Y%j"),
    "YYYYDDD_end": lambda x: pd.to_datetime(str(x), format="%Y%j"),
}

# Columns in the csv files to rename for consistency
_RENAME_COLUMNS = {
    "YYYYDDD": "Date",
    "#YYYYDDD": "Date",
    "#Comments": "Comments",
    "YYYYDDD_strt": "StartDate",
    "YYYYDDD_end": "EndDate",
}


def read_ancillary_file(ancillary_file: str | Path) -> pd.DataFrame:
    """
    Read a generic ancillary CSV file into a pandas DataFrame.

    Parameters
    ----------
    ancillary_file : str or Path
        Path to the ancillary CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ancillary data.
    """
    legacy_format = False
    read_csv_kwargs: dict[str, Any] = {}
    if "esa-mode-lut" in str(ancillary_file):
        # skip the first row which is a comment
        read_csv_kwargs["skiprows"] = [0]
    elif "geometric-factor" in str(ancillary_file):
        # legacy format - rows with comment headers indicating Hi_Res and Hi_Thr
        legacy_format = "Hi_Thr,,," in Path(ancillary_file).read_text()
        if legacy_format:
            read_csv_kwargs["skiprows"] = [1, 38]
        else:
            read_csv_kwargs["comment"] = "#"
    df = pd.read_csv(ancillary_file, converters=_CONVERTERS, **read_csv_kwargs)
    df = df.rename(columns=_RENAME_COLUMNS)

    if "geometric-factor" in str(ancillary_file):
        if legacy_format and "esa_mode" not in df.columns:
            # Add an ESA mode column based on the known structure of the file.
            # The first 36 rows are ESA mode 0 (HiRes), the second 36 are ESA mode 1
            # (HiThr)
            df["esa_mode"] = 0
            df.loc[36:, "esa_mode"] = 1

    return df
