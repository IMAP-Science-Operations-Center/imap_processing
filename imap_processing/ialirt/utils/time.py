"""Common time functions for I-ALiRT instruments."""

import xarray as xr


def calculate_time(
    sc_sclk_sec: xr.DataArray, sc_sclk_sub_sec: xr.DataArray, conversion: int
) -> xr.DataArray:
    """
    Calculate the time.

    Parameters
    ----------
    sc_sclk_sec : xr.DataArray
        SCLK seconds.
    sc_sclk_sub_sec : xr.DataArray
        SCLK subseconds.
    conversion : int
        Fine time units = 1 second.

    Returns
    -------
    time_seconds: xr.DataArray
        Calculated time.
    """
    time_fraction = sc_sclk_sub_sec / conversion
    time_seconds = sc_sclk_sec + time_fraction

    return time_seconds
