"""Functions for retrieving spin-table data."""

import functools
import logging
from functools import reduce
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy import typing as npt

from imap_processing.spice import config
from imap_processing.spice.geometry import (
    SpiceFrame,
    get_spacecraft_to_instrument_spin_phase_offset,
)
from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)


def set_global_spin_table_paths(paths: list[Path]) -> None:
    """
    Set the paths to input spin-table csv files.

    Parameters
    ----------
    paths : list[pathlib.Path]
        List of paths to spin-table csv files that will be used to supply
        spin-table data.
    """
    # If paths is an empty list, do nothing
    if not paths:
        return
    logger.info(
        f"Using the following spin-tables in processing: {[p.name for p in paths]}"
    )
    config._spin_table_paths = paths


def get_spin_data() -> pd.DataFrame:
    """
    Read spin-tables and return spin data.

    The spin-tables to read are stored in the mutable module level attribute
    named `spin_table_paths`.

    Returns
    -------
    spin_data : pandas.DataFrame
        Spin data. The DataFrame will have the following columns:

            * `spin_number`: Unique integer spin number.
            * `spin_start_sec_sclk`: MET seconds of spin start time.
            * `spin_start_subsec_sclk`: MET microseconds of spin start time.
            * `spin_start_met`: Floating point MET seconds of spin start.
            * `spin_start_utc`: UTC string of spin start time.
            * `spin_period_sec`: Floating point spin period in seconds.
            * `spin_period_valid`: Boolean indicating whether spin period is valid.
            * `spin_phase_valid`: Boolean indicating whether spin phase is valid.
            * `spin_period_source`: Source used for determining spin period.
            * `thruster_firing`: Boolean indicating whether thruster is firing.

    Raises
    ------
    ValueError
        If no spin-table paths have been set.
    """
    if config._spin_table_paths is None or len(config._spin_table_paths) == 0:
        # Handle the case where the module attribute is not set
        raise ValueError(
            "Spin-table paths have not been defined in spin.py "
            "module attribute spin_table_paths."
        )

    return _load_spin_data_with_cache(tuple(config._spin_table_paths))


@functools.cache
def _load_spin_data_with_cache(csv_paths: tuple[Path]) -> pd.DataFrame:
    """
    Load spin-table data from csv files and combine them.

    Parameters
    ----------
    csv_paths : tuple[Path]
        Locations of spin-table csv files.

    Returns
    -------
    combined_df: pandas.DataFrame
        The dataframe containing all spin data.
    """
    logger.debug(
        f"Merging the following spin tables files: {[sp.name for sp in csv_paths]}"
    )

    spin_dataframes = [
        pd.read_csv(
            spin_table_path,
            comment="#",
            index_col="spin_number",
            dtype={
                "spin_number": int,
                "spin_start_sec_sclk": int,
                "spin_start_subsec_sclk": int,
                # "spin_start_utc": str,
                "spin_period_sec": float,
                "spin_period_valid": bool,
                "spin_period_source": int,
                "thruster_firing": bool,
            },
        )
        # Reversed sorting is used so that the proper precedence is applied in
        # the below use of DataFrame.combine_first()
        for spin_table_path in sorted(csv_paths, reverse=True)
    ]
    combined_df = reduce(
        lambda left, right: left.combine_first(right),
        spin_dataframes,
    )
    # Duplicate the index so that users can access "spin_numer" by name
    combined_df.insert(0, "spin_number", combined_df.index)
    # Combine spin_start_sec_sclk and spin_start_subsec_sclk to get the spin start
    # time in seconds. The spin start subseconds are in microseconds.
    combined_df["spin_start_met"] = (
        combined_df["spin_start_sec_sclk"] + combined_df["spin_start_subsec_sclk"] / 1e6
    )
    # Convert MET to TTJ2000ns for direct use with MAG data
    combined_df["spin_start_ttj2000ns"] = met_to_ttj2000ns(
        combined_df["spin_start_met"]
    )
    return combined_df


def interpolate_spin_data(
    query_times: Union[float, npt.NDArray], time_format: str = "met"
) -> pd.DataFrame:
    """
    Interpolate spin table data to the queried times.

    All columns in the spin table csv file are interpolated to the previous
    table entry. A sc_spin_phase column is added that is the computed spacecraft
    spin phase at the queried times. Note that spin phase is by definition,
    in the interval [0, 1) where 1 is equivalent to 360 degrees.

    Parameters
    ----------
    query_times : float or np.ndarray
        Query times in the specified format.
    time_format : str
        Time format of query_times. Either "met" for Mission Elapsed Time
        or "j2000ns" for nanoseconds since J2000.

    Returns
    -------
    spin_df : pandas.DataFrame
        Spin table data interpolated for each queried time. In addition to
        the columns output from :py:func:`get_spin_data`, the `sc_spin_phase`
        column is added and is uniquely computed for each queried time.
    """
    spin_df = get_spin_data()

    # Ensure query_times is an array
    query_times = np.asarray(query_times)
    is_scalar = query_times.ndim == 0
    if is_scalar:
        # Force scalar to array because np.asarray() will not
        # convert scalar to array
        query_times = np.atleast_1d(query_times)

    # Select appropriate time column based on format
    if time_format == "j2000ns":
        spin_time_col = "spin_start_ttj2000ns"
        query_spin_times = query_times
    elif time_format == "met":
        spin_time_col = "spin_start_met"
        query_spin_times = query_times
    else:
        raise ValueError(
            f"Unsupported time_format: {time_format}. Use 'met' or 'j2000ns'."
        )

    # Make sure input times are within the bounds of spin data
    spin_df_start_time = spin_df[spin_time_col].values[0]
    spin_df_end_time = (
        spin_df[spin_time_col].values[-1] + spin_df["spin_period_sec"].values[-1] * 1e9
        if time_format == "j2000ns"
        else spin_df[spin_time_col].values[-1] + spin_df["spin_period_sec"].values[-1]
    )
    input_start_time = query_spin_times.min()
    input_end_time = query_spin_times.max()
    if input_start_time < spin_df_start_time or input_end_time >= spin_df_end_time:
        raise ValueError(
            f"Query times, {query_times} are outside of the spin data range, "
            f"{spin_df_start_time, spin_df_end_time}."
        )

    # Find all spin time that are less or equal to query_times.
    # To do that, use side right, a[i-1] <= v < a[i], in the searchsorted.
    # Eg.
    # >>> df['a']
    # array([0, 15, 30, 45, 60])
    # >>> np.searchsorted(df['a'], [0, 13, 15, 32, 70], side='right')
    # array([1, 1, 2, 3, 5])
    last_spin_indices = (
        np.searchsorted(spin_df[spin_time_col], query_spin_times, side="right") - 1
    )
    # Generate a dataframe with one row per query time
    out_df = spin_df.iloc[last_spin_indices]

    # Calculate spin phase
    if time_format == "j2000ns":
        # Convert spin period to nanoseconds and calculate phase
        spin_phases = (query_spin_times - out_df[spin_time_col].values) / (
            out_df["spin_period_sec"].values * 1e9
        )
    else:
        # MET time calculation (original logic)
        spin_phases = (query_spin_times - out_df[spin_time_col].values) / out_df[
            "spin_period_sec"
        ].values

    # Check for invalid spin phase using below checks:
    # 1. Check that the spin phase is in valid range, [0, 1).
    # 2. Check invalid spin phase using spin_phase_valid,
    #   spin_period_valid columns.
    invalid_spin_phase_range = (spin_phases < 0) | (spin_phases >= 1)

    invalid_spins = (out_df["spin_phase_valid"].values == 0) | (
        out_df["spin_period_valid"].values == 0
    )
    bad_spin_phases = invalid_spin_phase_range | invalid_spins
    spin_phases[bad_spin_phases] = np.nan

    # Add spin_phase column to output dataframe
    out_df["sc_spin_phase"] = spin_phases

    return out_df


def get_spin_angle(
    spin_phases: Union[float, npt.NDArray],
    degrees: bool = False,
) -> Union[float, npt.NDArray]:
    """
    Convert spin_phases to radians or degrees.

    Parameters
    ----------
    spin_phases : float or np.ndarray
        Instrument or spacecraft spin phases. Spin phase is a
        floating point number in the range [0, 1) corresponding to the
        spin angle / 360.
    degrees : bool
        If degrees parameter is True, return angle in degrees otherwise return angle in
        radians. Default is False.

    Returns
    -------
    spin_phases : float or np.ndarray
        Spin angle in degrees or radians for the input query times.
    """
    if np.any(spin_phases < 0) or np.any(spin_phases >= 1):
        raise ValueError(
            f"Spin phases, {spin_phases} are outside of the expected spin phase range, "
            f"[0, 1) "
        )
    if degrees:
        # Convert to degrees
        return spin_phases * 360
    else:
        # Convert to radians
        return spin_phases * 2 * np.pi


def get_spacecraft_spin_phase(
    query_met_times: Union[float, npt.NDArray],
) -> Union[float, npt.NDArray]:
    """
    Get the spacecraft spin phase for the input query times.

    Formula to calculate spin phase:
        spin_phase = (query_met_times - spin_start_met) / spin_period_sec

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).

    Returns
    -------
    spin_phase : float or np.ndarray
        Spin phase for the input query times.
    """
    spin_df = interpolate_spin_data(query_met_times, time_format="met")
    if np.asarray(query_met_times).ndim == 0:
        return spin_df["sc_spin_phase"].values[0]
    return spin_df["sc_spin_phase"].values


def get_spacecraft_spin_phase_j2000ns(
    query_j2000ns_times: Union[float, npt.NDArray],
) -> Union[float, npt.NDArray]:
    """
    Get the spacecraft spin phase for the input query times in J2000ns format.

    Formula to calculate spin phase:
        spin_phase = (query_j2000ns_times - spin_start_ttj2000ns) /
        (spin_period_sec * 1e9)

    Parameters
    ----------
    query_j2000ns_times : float or np.ndarray
        Query times in nanoseconds since J2000.

    Returns
    -------
    spin_phase : float or np.ndarray
        Spin phase for the input query times.
    """
    spin_df = interpolate_spin_data(query_j2000ns_times, time_format="j2000ns")
    if np.asarray(query_j2000ns_times).ndim == 0:
        return spin_df["sc_spin_phase"].values[0]
    return spin_df["sc_spin_phase"].values


def get_instrument_spin_phase(
    query_met_times: Union[float, npt.NDArray], instrument: SpiceFrame
) -> Union[float, npt.NDArray]:
    """
    Get the instrument spin phase for the input query times.

    Formula to calculate spin phase:
        instrument_spin_phase = (spacecraft_spin_phase + instrument_spin_offset) % 1

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).
    instrument : SpiceFrame
        Instrument frame to calculate spin phase for.

    Returns
    -------
    spin_phase : float or np.ndarray
        Instrument spin phase for the input query times. Spin phase is a
        floating point number in the range [0, 1) corresponding to the
        spin angle / 360.
    """
    spacecraft_spin_phase = get_spacecraft_spin_phase(query_met_times)
    instrument_spin_phase_offset = get_spacecraft_to_instrument_spin_phase_offset(
        instrument
    )
    return (spacecraft_spin_phase + instrument_spin_phase_offset) % 1
