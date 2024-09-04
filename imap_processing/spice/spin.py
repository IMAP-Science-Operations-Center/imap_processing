"""Utils function for generating and query spin and repoint table."""

from typing import Optional

import numpy as np
import pandas as pd


def generate_spin_table(start_met: int, end_met: Optional[int]) -> dict:
    """
    Generate a spin table CSV covering one or more days.

    Parameters
    ----------
    start_met : int
        Provides the start time in Mission Elapsed Time (MET).
    end_met : int
        Provides the end time in MET. If not provided, default to one day
        from start time.

    Returns
    -------
    spin_df : dict
        Spin data. May need to save data to CSV file.
    """
    if end_met is None:
        # end_time is one day after start_time
        end_met = start_met + 86400

    # TODO: Replace all of below code with actual spin data.
    # This is a temporary code to generate spin data for testing
    # and development purposes. In the future, this function will query
    # and return all spin data for the input date range.

    # Create spin start second data of 15 seconds increment
    spin_start_sec = np.arange(start_met, end_met, 15)

    # Spin table contains the following fields:
    # (
    #   spin_number,
    #   spin_start_sec,
    #   spin_start_subsec,
    #   spin_period_sec,
    #   spin_period_valid,
    #   spin_phas_valid,
    #   spin_period_source,
    #   thruster_firing
    # )
    spin_dict = {
        "spin_number": np.arange(spin_start_sec.size, dtype=np.uint32),
        "spin_start_sec": spin_start_sec,
        "spin_start_subsec": np.full(spin_start_sec.size, 0, dtype=np.uint32),
        "spin_period_sec": np.full(spin_start_sec.size, 15.0, dtype=np.float32),
        "spin_period_valid": np.ones(spin_start_sec.size, dtype=np.uint8),
        "spin_phas_valid": np.ones(spin_start_sec.size, dtype=np.uint8),
        "spin_period_source": np.zeros(spin_start_sec.size, dtype=np.uint8),
        "thruster_firing": np.zeros(spin_start_sec.size, dtype=np.uint8),
    }

    # Convert spin_start_sec to datetime to set repointing times
    spin_start_dates = pd.to_datetime(spin_dict["spin_start_sec"], unit="s")

    # Convert DatetimeIndex to Series for using .dt accessor
    spin_start_dates_series = pd.Series(spin_start_dates)

    # Find index of all timestamps that fall within 10 minutes after midnight
    repointing_times = spin_start_dates_series[
        (spin_start_dates_series.dt.time >= pd.Timestamp("00:00:00").time())
        & (spin_start_dates_series.dt.time < pd.Timestamp("00:10:00").time())
    ]

    repointing_times_index = repointing_times.index

    # Use the repointing times to set thruster firing flag and spin period valid
    spin_dict["thruster_firing"][repointing_times_index] = 1
    spin_dict["spin_period_valid"][repointing_times_index] = 0
    spin_dict["spin_phas_valid"][repointing_times_index] = 0

    return spin_dict
