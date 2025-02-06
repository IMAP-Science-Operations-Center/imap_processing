"""Functions for retrieving spin-table data."""

import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy import typing as npt

from imap_processing.spice.geometry import (
    SpiceFrame,
    get_spacecraft_to_instrument_spin_phase_offset,
)


def get_spin_data() -> pd.DataFrame:
    """
    Read spin file using environment variable and return spin data.

    SPIN_DATA_FILEPATH environment variable would be a fixed value.
    It could be s3 filepath that can be used to download the data
    through API or it could be path EFS or Batch volume mount path.

    Spin data should contain the following fields:
        * spin_number
        * spin_start_sec
        * spin_start_subsec
        * spin_period_sec
        * spin_period_valid
        * spin_phase_valid
        * spin_period_source
        * thruster_firing

    Returns
    -------
    spin_data : pandas.DataFrame
        Spin data.
    """
    spin_data_filepath = os.getenv("SPIN_DATA_FILEPATH")
    if spin_data_filepath is not None:
        path_to_spin_file = Path(spin_data_filepath)
    else:
        # Handle the case where the environment variable is not set
        raise ValueError("SPIN_DATA_FILEPATH environment variable is not set.")

    spin_df = pd.read_csv(path_to_spin_file, comment="#")
    # Combine spin_start_sec and spin_start_subsec to get the spin start
    # time in seconds. The spin start subseconds are in milliseconds.
    spin_df["spin_start_time"] = (
        spin_df["spin_start_sec"] + spin_df["spin_start_subsec"] / 1e3
    )

    return spin_df


def interpolate_spin_data(query_met_times: Union[float, npt.NDArray]) -> pd.DataFrame:
    """
    Interpolate spin table data to the queried MET times.

    All columns in the spin table csv file are interpolated to the previous
    table entry. A sc_spin_phase column is added that is the computed spacecraft
    spin phase at the queried MET times. Note that spin phase is by definition,
    in the interval [0, 1) where 1 is equivalent to 360 degrees.

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).

    Returns
    -------
    spin_df : pandas.DataFrame
        Spin table data with the spin-phase column added and one row
        interpolated for each queried MET time. Output columns are:
        * spin_number
        * spin_start_sec
        * spin_start_subsec
        * spin_period_sec
        * spin_period_valid
        * spin_phase_valid
        * spin_period_source
        * thruster_firing
        * spin_start_met
        * sc_spin_phase
    """
    spin_df = get_spin_data()

    # Ensure query_met_times is an array
    query_met_times = np.asarray(query_met_times)
    is_scalar = query_met_times.ndim == 0
    if is_scalar:
        # Force scalar to array because np.asarray() will not
        # convert scalar to array
        query_met_times = np.atleast_1d(query_met_times)

    # Make sure input times are within the bounds of spin data
    spin_df_start_time = spin_df["spin_start_time"].values[0]
    spin_df_end_time = (
        spin_df["spin_start_time"].values[-1] + spin_df["spin_period_sec"].values[-1]
    )
    input_start_time = query_met_times.min()
    input_end_time = query_met_times.max()
    if input_start_time < spin_df_start_time or input_end_time >= spin_df_end_time:
        raise ValueError(
            f"Query times, {query_met_times} are outside of the spin data range, "
            f"{spin_df_start_time, spin_df_end_time}."
        )

    # Find all spin time that are less or equal to query_met_times.
    # To do that, use side right, a[i-1] <= v < a[i], in the searchsorted.
    # Eg.
    # >>> df['a']
    # array([0, 15, 30, 45, 60])
    # >>> np.searchsorted(df['a'], [0, 13, 15, 32, 70], side='right')
    # array([1, 1, 2, 3, 5])
    last_spin_indices = (
        np.searchsorted(spin_df["spin_start_time"], query_met_times, side="right") - 1
    )
    # Generate a dataframe with one row per query time
    out_df = spin_df.iloc[last_spin_indices]

    # Calculate spin phase
    spin_phases = (query_met_times - out_df["spin_start_time"].values) / out_df[
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
        spin_phase = (query_met_times - spin_start_time) / spin_period_sec

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).

    Returns
    -------
    spin_phase : float or np.ndarray
        Spin phase for the input query times.
    """
    spin_df = interpolate_spin_data(query_met_times)
    if np.asarray(query_met_times).ndim == 0:
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
