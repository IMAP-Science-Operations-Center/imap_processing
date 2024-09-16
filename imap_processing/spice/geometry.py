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
from typing import Union

import numpy as np
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
