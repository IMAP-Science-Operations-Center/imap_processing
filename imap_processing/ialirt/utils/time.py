"""Common time functions for I-ALiRT instruments."""

import xarray as xr


def calculate_time(
    coarse_time: xr.DataArray, fin_time: xr.DataArray, conversion: int
) -> xr.DataArray:
    """
    Calculate the time.

    Parameters
    ----------
    coarse_time : xr.DataArray
        Coarse time.
    fin_time : xr.DataArray
        Fine time.
    conversion : int
        Fine time units = 1 second.

    Returns
    -------
    time_seconds: xr.DataArray
        Calculated time.
    """
    fine_time_fraction = fin_time / conversion
    time_seconds = coarse_time + fine_time_fraction

    return time_seconds
