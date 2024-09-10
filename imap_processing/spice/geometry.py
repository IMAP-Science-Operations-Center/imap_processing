"""
Functions for computing geometry, many of which use SPICE.

Paradigms for developing this module:
* Use @ensure_spice decorator on functions that directly wrap spiceypy functions
* Vectorize everything at the lowest level possible (e.g. the decorated spiceypy
  wrapper function)
* Always return numpy arrays for vectorized calls.
"""

import typing
from enum import IntEnum
from typing import Optional, Union

import cdflib
import numpy as np
import pandas as pd
import spiceypy as spice

from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.spice.kernels import ensure_spice


class SpiceBody(IntEnum):
    """Enum containing SPICE IDs for bodies that we use."""

    # A subset of IMAP Specific bodies as defined in imap_wkcp.tf
    IMAP = -43
    IMAP_SPACECRAFT = -43000
    # IMAP Pointing Frame (Despun) as defined in iamp_science_0001.tf
    IMAP_DPS = -43901
    # Standard NAIF bodies
    SOLAR_SYSTEM_BARYCENTER = spice.bodn2c("SOLAR_SYSTEM_BARYCENTER")
    SUN = spice.bodn2c("SUN")
    EARTH = spice.bodn2c("EARTH")


class SpiceFrame(IntEnum):
    """Enum containing SPICE IDs for reference frames, defined in imap_wkcp.tf."""

    # Standard SPICE Frames
    J2000 = spice.irfnum("J2000")
    ECLIPJ2000 = spice.irfnum("ECLIPJ2000")
    # IMAP specific as defined in imap_wkcp.tf
    IMAP_SPACECRAFT = -43000
    IMAP_LO_BASE = -43100
    IMAP_LO_STAR_SENSOR = -43103
    IMAP_LO = -43105
    IMAP_HI_45 = -43150
    IMAP_HI_90 = -43160
    IMAP_ULTRA_45 = -43200
    IMAP_ULTRA_90 = -43210
    IMAP_MAG = -43250
    IMAP_SWE = -43300
    IMAP_SWAPI = -43350
    IMAP_CODICE = -43400
    IMAP_HIT = -43500
    IMAP_IDEX = -43700
    IMAP_GLOWS = -43750


@typing.no_type_check
@ensure_spice
def imap_state(
    et: Union[np.ndarray, float],
    ref_frame: SpiceFrame = SpiceFrame.ECLIPJ2000,
    observer: SpiceBody = SpiceBody.SUN,
) -> np.ndarray:
    """
    Get the state (position and velocity) of the IMAP spacecraft.

    By default, the state is returned in the ECLIPJ2000 frame as observed by the Sun.

    Parameters
    ----------
    et : np.ndarray or float
        Epoch time(s) [J2000 seconds] to get the IMAP state for.
    ref_frame : SpiceFrame, optional
        Reference frame which the IMAP state is expressed in.
    observer : SpiceBody, optional
        Observing body.

    Returns
    -------
    state : np.ndarray
     The Cartesian state vector representing the position and velocity of the
     IMAP spacecraft.
    """
    state, _ = spice.spkezr(
        SpiceBody.IMAP.name, et, ref_frame.name, "NONE", observer.name
    )
    return np.asarray(state)


def generate_spin_data(start_met: int, end_met: Optional[int] = None) -> dict:
    """
    Generate a spin table CSV covering one or more days.

    Spin table contains the following fields:
        (
        spin_number,
        spin_start_sec,
        spin_start_subsec,
        spin_period_sec,
        spin_period_valid,
        spin_phas_valid,
        spin_period_source,
        thruster_firing
        )
    This function creates spin data using start MET and end MET time.
    Each spin start data uses the nominal 15 second spin period. The spins that
    occur from 00:00(Mid-night) to 00:10 UTC are marked with flags for
    thruster firing, invalid spin period, and invalid spin phase.

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

    # Create spin start second data of 15 seconds increment
    spin_start_sec = np.arange(start_met, end_met, 15)

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

    # Convert spin_start_sec to datetime to set repointing times flags
    spin_start_dates = met_to_j2000ns(spin_start_sec)
    spin_start_dates = cdflib.cdfepoch.to_datetime(spin_start_dates)

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


def get_spin_data(start_met: int, end_met: Optional[int] = None) -> dict:
    """
    Get spin data for a given time range.

    This function queries spin data for the input date range. Spin
    table contains the following fields:
        (
            spin_number,
            spin_start_sec,
            spin_start_subsec,
            spin_period_sec,
            spin_period_valid,
            spin_phas_valid,
            spin_period_source,
            thruster_firing
        )

    Parameters
    ----------
    start_met : int
        Provide the start time in Mission Elapsed Time (MET).
    end_met : int
        Provide the end time in MET. If not provided, default to one day
        from start time.

    Returns
    -------
    spin_data : dict
        Spin data. It's a dictionary with keys and values as numpy arrays.
    """
    if end_met is None:
        # end time is one day after start time
        end_met = start_met + 86400

    # TODO: write code to query spin database and return all spin data
    # for the input date range once we have actual infrastructure in place.

    # Call generate_spin_data function temporarily that generates sample spin data for
    # the given date range.
    return generate_spin_data(start_met, end_met)


def get_spacecraft_spin_phase(
    query_times: Union[float, np.ndarray], spin_data: dict
) -> Union[float, np.ndarray]:
    """
    Get the spacecraft spin phase for the input query times.

    Formula to calculate spin phase:
        spin_phase = (
            query_times - (spin_start_seconds + spin_start_subseconds)
        ) / spin_period_sec

    Parameters
    ----------
    query_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).
    spin_data : dict
        Spin data for the spacecraft.

    Returns
    -------
    spin_phase : float or np.ndarray
        Spin phase for the input query times.
    """
    # TODO: Write code to calculate spin phase for the input query times
    if isinstance(query_times, float):
        query_times = np.asarray([query_times])
    return query_times
