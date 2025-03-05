"""Functions for retrieving repointing table data."""

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy import typing as npt

logger = logging.getLogger(__name__)


def get_repoint_data() -> pd.DataFrame:
    """
    Read repointing file using environment variable and return as dataframe.

    REPOINT_DATA_FILEPATH environment variable should point to a local
    file where the repointing csv file is located.

    Returns
    -------
    repoint_df : pd.DataFrame
        The repointing csv loaded into a pandas dataframe. The dataframe will
        contain the following columns:
            - `repoint_start_time`: Starting MET time of each repoint maneuver.
            - `repoint_end_time`: Ending MET time of each repoint maneuver.
            - `repoint_id`: Unique ID number of each repoint maneuver.
    """
    repoint_data_filepath = os.getenv("REPOINT_DATA_FILEPATH")
    if repoint_data_filepath is not None:
        path_to_spin_file = Path(repoint_data_filepath)
    else:
        # Handle the case where the environment variable is not set
        raise ValueError("REPOINT_DATA_FILEPATH environment variable is not set.")

    logger.info(f"Reading repointing data from {path_to_spin_file}")
    repoint_df = pd.read_csv(path_to_spin_file, comment="#")

    return repoint_df


def interpolate_repoint_data(
    query_met_times: Union[float, npt.NDArray],
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
            - `repoint_start_time`
            - `repoint_end_time`
            - `repoint_id`
            - `repoint_in_progress`

    Raises
    ------
    ValueError : If any of the query_met_times are before the first repoint
    start time or after the last repoint start time plus 24-hours.
    """
    repoint_df = get_repoint_data()

    # Ensure query_met_times is an array
    query_met_times = np.atleast_1d(query_met_times)

    # Make sure no query times are before the first repoint in the dataframe.
    repoint_df_start_time = repoint_df["repoint_start_time"].values[0]
    if np.any(query_met_times < repoint_df_start_time):
        bad_times = query_met_times[query_met_times < repoint_df_start_time]
        raise ValueError(
            f"{bad_times.size} query times are before the first repoint start "
            f" time in the repoint table. {bad_times=}, {repoint_df_start_time=}"
        )
    # Make sure that no query times are after the valid range of the dataframe.
    # We approximate the end time of the table by adding 24 hours to the last
    # known repoint start time.
    repoint_df_end_time = repoint_df["repoint_start_time"].values[-1] + 24 * 60 * 60
    if np.any(query_met_times >= repoint_df_end_time):
        bad_times = query_met_times[query_met_times >= repoint_df_end_time]
        raise ValueError(
            f"{bad_times.size} query times are after the valid time of the "
            f"pointing table. The valid end time is 24-hours after the last "
            f"repoint_start_time. {bad_times=}, {repoint_df_end_time=}"
        )

    # Find the row index for each queried MET time such that:
    # repoint_start_time[i] <= MET < repoint_start_time[i+1]
    row_indices = (
        np.searchsorted(repoint_df["repoint_start_time"], query_met_times, side="right")
        - 1
    )
    out_df = repoint_df.iloc[row_indices]

    # Add a column indicating if the query time is during a repoint or not.
    # The table already has the correct row for each query time, so we
    # only need to check if the query time is less than the repoint end time to
    # get the same result as `repoint_start_time <= query_met_times < repoint_end_time`.
    out_df["repoint_in_progress"] = query_met_times < out_df["repoint_end_time"].values

    return out_df
