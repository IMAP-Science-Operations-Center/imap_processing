"""Functions for retrieving repointing table data."""

import functools
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import typing as npt

from imap_processing.spice import config
from imap_processing.spice.geometry import imap_state
from imap_processing.spice.time import met_to_sclkticks, sct_to_et

logger = logging.getLogger(__name__)


def set_global_repoint_table_paths(paths: list[Path]) -> None:
    """
    Set the path to input repoint-table csv file.

    Parameters
    ----------
    paths : list[pathlib.Path]
        List of paths to repoint-table csv files that will be used to supply
        repoint-table data. Note that although a list of Path objects is allowed,
        only a list of length 0 or 1 is supported.

    Raises
    ------
    ValueError
        If paths contains more than one repoint-table csv file path.
    """
    # If paths is an empty list, do nothing
    if not paths:
        return
    elif len(paths) > 1:
        raise ValueError("Cannot set repoint-table paths to more than one file.")
    logger.info(f"Using the following repoint table in processing: {paths[0].name}")
    config._repoint_table_path = paths[0]


def get_repoint_data() -> pd.DataFrame:
    """
    Read repointing file from the configured location and return as dataframe.

    Pointing and repointing nomenclature can be confusing. In this case,
    repoint is taken to mean a repoint maneuver. Thus, repoint_start and repoint_end
    are the times that bound when the spacecraft is performing a repointing maneuver.
    This is different from a pointing which is the time between repointing maneuvers.

    The repoint table location is stored in the global variable `_repoint_table_path`
    in the imap_processing.spice.config module.

    Returns
    -------
    repoint_df : pandas.DataFrame
        The repointing csv loaded into a pandas dataframe. The dataframe will
        contain the following columns:

            * `repoint_start_sec_sclk`: Starting MET seconds of repoint maneuver.
            * `repoint_start_subsec_sclk`: Starting MET microseconds of repoint
              maneuver.
            * `repoint_start_met`: Floating point MET of repoint maneuver start time.
              Derived from `repoint_start_sec_sclk` and `repoint_start_subsec_sclk`.
            * `repoint_start_utc`: UTC time of repoint maneuver start time.
            * `repoint_end_sec_sclk`: Ending MET seconds of repoint maneuver.
            * `repoint_end_subsec_sclk`: Ending MET microseconds of repoint maneuver.
            * `repoint_end_met`: Floating point MET of repoint maneuver end time.
              Derived from `repoint_end_sec_sclk` and `repoint_end_subsec_sclk`.
            * `repoint_end_utc`: UTC time of repoint maneuver end time.
            * `repoint_id`: Unique ID number of each repoint maneuver.

    Raises
    ------
    ValueError
        If no path to a repoint-table has been set.
    """
    if config._repoint_table_path is None:
        raise ValueError(
            "No repoint-table path as been defined in repoint.py "
            "module attribute repoint_table_path."
        )
    return _load_repoint_data_with_cache(config._repoint_table_path)


@functools.cache
def _load_repoint_data_with_cache(csv_path: Path) -> pd.DataFrame:
    """
    Load repointing data from csv file.

    Parameters
    ----------
    csv_path : Path
        Location of repointing csv file.

    Returns
    -------
    repoint_df : pandas.DataFrame
        See `get_repoint_data` documentation regarding dataframe contents.
    """
    logger.debug(f"Reading in the following repoint table file: {csv_path.name}")
    repoint_df = pd.read_csv(csv_path, comment="#")

    # Compute times by combining seconds and subseconds fields
    repoint_df["repoint_start_met"] = (
        repoint_df["repoint_start_sec_sclk"]
        + repoint_df["repoint_start_subsec_sclk"] / 1e6
    )
    repoint_df["repoint_end_met"] = (
        repoint_df["repoint_end_sec_sclk"] + repoint_df["repoint_end_subsec_sclk"] / 1e6
    )
    return repoint_df


def interpolate_repoint_data(
    query_met_times: float | npt.NDArray,
) -> pd.DataFrame:
    """
    Interpolate repointing data to the queried MET times.

    In addition to the repoint start, end, and id values that come directly from
    the universal repointing table, a column is added to the output dataframe
    which indicates whether each query met time occurs during a repoint maneuver
    i.e. between the repoint start and end times of a row in the repointing
    table.

    Query times that are more than 24-hours after that last repoint start time
    in the repoint table will cause an error to be raised. The assumption here
    is that we shouldn't be processing data that occurs that close to the next
    expected repoint start time before getting an updated repoint table.

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).

    Returns
    -------
    repoint_df : pandas.DataFrame
        Repoint table data interpolated such that there is one row
        for each of the queried MET times. Output columns are:

            * `repoint_start_sec_sclk`
            * `repoint_start_subsec_sclk`
            * `repoint_start_met`
            * `repoint_end_sec_sclk`
            * `repoint_end_subsec_sclk`
            * `repoint_end_met`
            * `repoint_id`
            * `repoint_in_progress`

    Raises
    ------
    ValueError : If any of the query_met_times are before the first repoint
        start time or after the last repoint start time plus 24-hours.
    """
    repoint_df = get_repoint_data()

    # Ensure query_met_times is an array
    query_met_times = np.atleast_1d(query_met_times)

    # Make sure no query times are before the first repoint in the dataframe.
    repoint_df_start_met = repoint_df["repoint_start_met"].values[0]
    if np.any(query_met_times < repoint_df_start_met):
        bad_times = query_met_times[query_met_times < repoint_df_start_met]
        raise ValueError(
            f"{bad_times.size} query times are before the first repoint start "
            f" time in the repoint table. {bad_times=}, {repoint_df_start_met=}"
        )
    # Make sure that no query times are after the valid range of the dataframe.
    # We approximate the end time of the table by adding 24 hours to the last
    # known repoint start time.
    repoint_df_end_met = repoint_df["repoint_start_met"].values[-1] + 24 * 60 * 60
    if np.any(query_met_times >= repoint_df_end_met):
        bad_times = query_met_times[query_met_times >= repoint_df_end_met]
        raise ValueError(
            f"{bad_times.size} query times are after the valid time of the "
            f"pointing table. The valid end time is 24-hours after the last "
            f"repoint_start_time. {bad_times=}, {repoint_df_end_met=}"
        )

    # Find the row index for each queried MET time such that:
    # repoint_start_time[i] <= MET < repoint_start_time[i+1]
    row_indices = (
        np.searchsorted(repoint_df["repoint_start_met"], query_met_times, side="right")
        - 1
    )
    out_df = repoint_df.iloc[row_indices]

    # Add a column indicating if the query time is during a repoint or not.
    # The table already has the correct row for each query time, so we
    # only need to check if the query time is less than the repoint end time to
    # get the same result as `repoint_start_time <= query_met_times < repoint_end_time`.
    out_df["repoint_in_progress"] = query_met_times < out_df["repoint_end_met"].values

    return out_df


def get_pointing_times(met_time: float) -> tuple[float, float]:
    """
    Get the start and end MET times for the pointing that contains the query MET time.

    Parameters
    ----------
    met_time : float
        The MET time in a pointing.

    Returns
    -------
    pointing_start_time : float
        The MET time of the repoint maneuver that ends before the query MET time.
    pointing_end_time : float
        The MET time of the repoint maneuver that starts after the query MET time.
    """
    # Find the pointing start time by finding the repoint end time
    repoint_df = interpolate_repoint_data(met_time)
    pointing_start_met = repoint_df["repoint_end_met"].item()
    # Find the pointing end time by finding the next repoint start time
    repoint_df = get_repoint_data()
    pointing_idx = repoint_df.index[
        repoint_df["repoint_end_met"] == pointing_start_met
    ][0]
    pointing_end_met = repoint_df["repoint_start_met"].iloc[pointing_idx + 1].item()
    return pointing_start_met, pointing_end_met


def get_pointing_times_from_id(repoint_id: int | str) -> tuple[float, float]:
    """
    Get the start and end MET times for the pointing given a repoint ID.

    Parameters
    ----------
    repoint_id : int
        The repoint ID corresponding to the pointing.

    Returns
    -------
    pointing_start_time : float
        The MET time of the repoint maneuver that ends before the query MET time.
    pointing_end_time : float
        The MET time of the repoint maneuver that starts after the query MET time.
    """
    if isinstance(repoint_id, str):
        if not bool(re.fullmatch(r"repoint\d{5}", str(repoint_id))):
            raise ValueError(
                f"Invalid repoint ID string format: {repoint_id}. "
                f"Expected format is 'repointXXXXX'"
            )

        repoint_id = int(repoint_id.replace("repoint", ""))

    repoint_df = get_repoint_data()
    # To find the pointing start and stop, get the end of the current repointing
    # and the start of the next repointing
    repoint_row = repoint_df[repoint_df["repoint_id"] == repoint_id]
    if repoint_row.empty:
        raise ValueError(f"Repoint ID {repoint_id} not found in repoint table.")
    next_repoint_row = repoint_df[repoint_df["repoint_id"] == repoint_id + 1]
    if next_repoint_row.empty:
        raise ValueError(
            f"Pointing end time not found for repoint ID {repoint_id}. Either current "
            "pointing is ongoing or the repoint table is outdated."
        )

    pointing_start_met = repoint_row["repoint_end_met"].values[0]
    pointing_end_met = next_repoint_row["repoint_start_met"].values[0]
    return pointing_start_met, pointing_end_met


def get_pointing_mid_time(met_time: float) -> float:
    """
    Get mid-point of the pointing for the given MET time.

    Get the mid-point time between the end of one repoint and
    start of the next. Input could be a MET time.

    Parameters
    ----------
    met_time : float
        The MET time in a repoint.

    Returns
    -------
    repoint_mid_time : float
        The mid MET time of the repoint maneuver.
    """
    pointing_start_met, pointing_end_met = get_pointing_times(met_time)
    return (pointing_start_met + pointing_end_met) / 2


def get_mid_point_state(met_time: float) -> npt.NDArray:
    """
    Get IMAP state for the mid-point.

    Get IMAP state for the mid-point of the pointing in
    reference frame, ECLIPJ2000 and observer, SUN.

    Parameters
    ----------
    met_time : float
        The MET time in a pointing.

    Returns
    -------
    mid_point_state : numpy.ndarray
        The mid state of the pointing maneuver.
    """
    # Get mid point time in ET
    mid_point_time = get_pointing_mid_time(met_time)
    mid_point_time_et = sct_to_et(met_to_sclkticks(mid_point_time))

    # Convert mid point time to state
    pointing_state = imap_state(mid_point_time_et)
    return pointing_state
