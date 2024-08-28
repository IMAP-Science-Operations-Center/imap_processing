"""Functions for computing geometry, many of which use SPICE."""

from collections.abc import Iterable
from enum import Enum
from typing import NamedTuple, Optional, Union

import numpy as np
import spiceypy as spice

from imap_processing.spice.kernels import ensure_spice


class SpiceId(NamedTuple):
    """Class that represents a unique identifier in the NAIF SPICE library."""

    strid: str
    numid: int


class SpiceBody(Enum):
    """Enum containing SPICE IDs for bodies that we use."""

    # A subset of IMAP Specific bodies as defined in imap_wkcp.tf
    IMAP = SpiceId("IMAP", -43)
    IMAP_SPACECRAFT = SpiceId("IMAP_SPACECRAFT", -43000)
    # IMAP Pointing Frame (Despun) as defined in iamp_science_0001.tf
    IMAP_DPS = SpiceId("IMAP_DPS", -43901)
    # Standard NAIF bodies
    SSB = SpiceId("SOLAR_SYSTEM_BARYCENTER", 0)
    SUN = SpiceId("SUN", 10)
    EARTH = SpiceId("EARTH", 399)


class SpiceFrame(Enum):
    """Enum containing SPICE IDs for reference frames, defined in imap_wkcp.tf."""

    # Standard SPICE Frames
    J2000 = SpiceId("J2000", 1)
    ECLIPJ2000 = SpiceId("ECLIPJ2000", 17)
    # IMAP specific
    IMAP_SPACECRAFT = SpiceId("IMAP_SPACECRAFT", -43000)
    IMAP_LO_BASE = SpiceId("IMAP_LO_BASE", -43100)
    IMAP_LO_STAR_SENSOR = SpiceId("IMAP_LO_STAR_SENSOR", -43103)
    IMAP_LO = SpiceId("IMAP_LO", -43105)
    IMAP_HI_45 = SpiceId("IMAP_HI_45", -43150)
    IMAP_HI_90 = SpiceId("IMAP_HI_90", -43160)
    IMAP_ULTRA_45 = SpiceId("IMAP_ULTRA_45", -43200)
    IMAP_ULTRA_90 = SpiceId("IMAP_ULTRA_90", -43210)
    IMAP_MAG = SpiceId("IMAP_MAG", -43250)
    IMAP_SWE = SpiceId("IMAP_SWE", -43300)
    IMAP_SWAPI = SpiceId("IMAP_SWAPI", -43350)
    IMAP_CODICE = SpiceId("IMAP_CODICE", -43400)
    IMAP_HIT = SpiceId("IMAP_HIT", -43500)
    IMAP_IDEX = SpiceId("IMAP_IDEX", -43700)
    IMAP_GLOWS = SpiceId("IMAP_GLOWS", -43750)


def imap_state(
    et: Union[np.ndarray, float],
    ref_frame: Optional[SpiceFrame] = None,
    observer: Optional[SpiceBody] = None,
) -> Union[np.ndarray, Iterable[np.ndarray]]:
    """
    Get the state (position and velocity) of the IMPA spacecraft.

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
    state : np.ndarray or Iterable[np.ndarray]
     The Cartesian state vector representing the position and velocity of the
     IMAP spacecraft.
    """
    if ref_frame is None:
        ref_frame = SpiceFrame.ECLIPJ2000
    if observer is None:
        observer = SpiceBody.SUN
    state, light_time = ensured_spkezr(
        SpiceBody.IMAP.name, et, ref_frame.name, "NONE", observer.name
    )
    return state


def ensured_spkezr(
    targ: str, et: Union[np.ndarray, float], ref: str, abcorr: str, obs: str
) -> Union[tuple[np.ndarray, float], tuple[Iterable[np.ndarray], Iterable[float]]]:
    """
    Wrap spice.spkezr() function with ensure_spice.

    Parameters
    ----------
    targ : str
        Target body name.
    et : ndarray or float
        J2000 observer times.
    ref : str
        Reference frame name.
    abcorr : str
        Aberration correction method.
    obs : str
        Observing body name.

    Returns
    -------
    state : np.ndarray or Iterable[np.ndarray]
        State of target.
    light_time : float or Iterable[float]
        One way light time between observer and target.

    Notes
    -----
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/spkezr.html
    """
    # No vectorization is needed b/c spiceypy already adds vectorization to the
    # spkezr function. If specific time coverage functionality is added to
    # @ensure_spice, parameters can be added here.
    ensured = ensure_spice(spice.spkezr)
    state, light_time = ensured(targ, et, ref, abcorr, obs)
    return state, light_time
