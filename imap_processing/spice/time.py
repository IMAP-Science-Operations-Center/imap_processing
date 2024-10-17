"""Time conversion functions that rely on SPICE."""

import typing
from collections.abc import Collection
from typing import Union

import numpy as np
import numpy.typing as npt
import spiceypy as spice

from imap_processing.spice import IMAP_SC_ID
from imap_processing.spice.kernels import ensure_spice

TICK_DURATION = 2e-5  # 20 microseconds as defined in imap_sclk_0000.tsc


def met_to_j2000ns(
    met: npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """
    Convert mission elapsed time (MET) to nanoseconds since J2000.

    Parameters
    ----------
    met : float, numpy.ndarray
        Number of seconds since epoch according to the spacecraft clock.

    Returns
    -------
    numpy.ndarray[numpy.int64]
        The mission elapsed time converted to nanoseconds since the J2000 epoch.

    Notes
    -----
    There are two options when using SPICE to convert from SCLK time (MET) to
    J2000. The conversion can be done on SCLK strings as input or using double
    precision continuous spacecraft clock "ticks". The latter is more accurate
    as it will correctly convert fractional clock ticks to nanoseconds. Since
    some IMAP instruments contain clocks with higher precision than 1 SCLK
    "tick" which is defined to be 20 microseconds, according to the sclk kernel,
    it is preferable to use the higher accuracy method.
    """
    sclk_ticks = np.asarray(met, dtype=float) / TICK_DURATION
    return np.asarray(_sct2e_wrapper(sclk_ticks) * 1e9, dtype=np.int64)


def j2000ns_to_j2000s(j2000ns: npt.ArrayLike) -> npt.NDArray[float]:
    """
    Convert the J2000 epoch nanoseconds to J2000 epoch seconds.

    The common CDF coordinate `epoch` stores J2000 nanoseconds. SPICE requires
    J2000 seconds be used. This is a common function to do that conversion.

    Parameters
    ----------
    j2000ns : float, numpy.ndarray
        Number of nanoseconds since the J2000 epoch.

    Returns
    -------
    numpy.ndarray[float]
        Number of seconds since the J2000 epoch.
    """
    return np.asarray(j2000ns, dtype=np.float64) / 1e9


@typing.no_type_check
@ensure_spice
def _sct2e_wrapper(
    sclk_ticks: Union[float, Collection[float]],
) -> Union[float, np.ndarray]:
    """
    Convert encoded spacecraft clock "ticks" to ephemeris time.

    Decorated wrapper for spiceypy.sct2e that vectorizes the function in addition
    to wrapping with the @ensure_spice automatic kernel furnishing functionality.
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/sct2e_c.html

    Parameters
    ----------
    sclk_ticks : Union[float, Collection[float]]
        Input sclk ticks value(s) to be converted to ephemeris time.

    Returns
    -------
    ephemeris_time: np.ndarray
        Ephemeris time, seconds past J2000.
    """
    if isinstance(sclk_ticks, Collection):
        return np.array([spice.sct2e(IMAP_SC_ID, s) for s in sclk_ticks])
    else:
        return spice.sct2e(IMAP_SC_ID, sclk_ticks)
