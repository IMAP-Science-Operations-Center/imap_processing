"""
Functions for computing geometry, many of which use SPICE.

Paradigms for developing this module:
* Use @ensure_spice decorator on functions that directly wrap spiceypy functions
* Vectorize everything at the lowest level possible (e.g. the decorated spiceypy
  wrapper function)
* Always return numpy arrays for vectorized calls.
"""

import os
import typing
from enum import IntEnum
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import spiceypy as spice

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


def get_spin_data() -> pd.DataFrame:
    """
    Read spin file using environment variable and return spin data.

    SPIN_DATA_FILEPATH environment variable would be a fixed value.
    It could be s3 filepath that can be used to download the data
    through API or it could be path EFS or Batch volume mount path.

    Spin data should contains the following fields:
        (
            spin_number,
            spin_start_sec,
            spin_start_subsec,
            spin_period_sec,
            spin_period_valid,
            spin_phase_valid,
            spin_period_source,
            thruster_firing
        )

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

    return pd.read_csv(path_to_spin_file)


def get_spacecraft_spin_phase(
    query_met_times: Union[float, npt.NDArray],
) -> Union[float, npt.NDArray]:
    """
    Get the spacecraft spin phase for the input query times.

    Formula to calculate spin phase:
        spin_phase = (
            query_met_times - (spin_start_seconds + spin_start_subseconds / 1e3)
        ) / spin_period_sec

    Parameters
    ----------
    query_met_times : float or np.ndarray
        Query times in Mission Elapsed Time (MET).

    Returns
    -------
    spin_phase : float or np.ndarray
        Spin phase for the input query times.
    """
    spin_df = get_spin_data()

    # Combine spin_start_sec and spin_start_subsec to get the spin start
    # time in seconds. The spin start subseconds are in milliseconds.
    # TODO: Decide if we should do this calculation in the data itself
    # or in get_spin_data function.
    spin_df["spin_start_time"] = (
        spin_df["spin_start_sec"] + spin_df["spin_start_subsec"] / 1e3
    )

    if isinstance(query_met_times, float):
        # calculate spin phase for a single query time
        mask = spin_df["spin_start_time"] <= query_met_times
        if mask.any():
            last_spin = spin_df[mask].iloc[-1]
            return (query_met_times - last_spin["spin_start_time"]) / last_spin[
                "spin_period_sec"
            ]
        else:
            raise ValueError(
                f"No spin data found for this data time: {query_met_times}"
            )

    # Create an empty array to store spin phase results
    spin_phases = np.zeros_like(query_met_times)

    # Loop through each query time and find the corresponding spin start time
    for i, data_time in enumerate(query_met_times):
        # Find the row where the spin start time is less than or
        # equal to the query time
        mask = spin_df["spin_start_time"] <= data_time

        if mask.any():
            # Get the last spin before or at the query time
            last_spin = spin_df[mask].iloc[-1]
            # Calculate spin phase using the formula
            spin_phases[i] = (data_time - last_spin["spin_start_time"]) / last_spin[
                "spin_period_sec"
            ]
        else:
            raise ValueError(f"No spin data found for this data time: {data_time}")
    return spin_phases
