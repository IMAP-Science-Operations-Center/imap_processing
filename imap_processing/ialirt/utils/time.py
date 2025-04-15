"""Common time functions for I-ALiRT instruments."""

import xarray as xr


def calculate_time(
    coarse_time: xr.DataArray, fine_time: xr.DataArray, conversion: int
) -> xr.DataArray:
    """
    Calculate the time.

    Parameters
    ----------
    coarse_time : xr.DataArray
        Coarse time.
    fine_time : xr.DataArray
        Fine time.
    conversion : int
        Fine time units = 1 second.

    Returns
    -------
    time_seconds: xr.DataArray
        Calculated time.
    """
    fine_time_fraction = fine_time / conversion
    time_seconds = coarse_time + fine_time_fraction

    return time_seconds
