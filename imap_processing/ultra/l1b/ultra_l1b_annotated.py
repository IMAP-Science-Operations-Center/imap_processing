"""Calculates Annotated Events for ULTRA L1b."""

import typing

import numpy as np
import spiceypy as spice
from numpy.typing import NDArray

from imap_processing.spice.kernels import ensure_spice


@ensure_spice
@typing.no_type_check
def get_particle_velocity(
    time: np.ndarray,
    instrument_velocity: np.ndarray,
    instrument_frame: str,
) -> NDArray[np.float64]:
    """
    Get the particle velocity in the pointing (DPS) frame wrt the spacecraft.

    Parameters
    ----------
    time : np.ndarray
        Ephemeris time.
    instrument_velocity : np.ndarray
        Particle velocity in the instrument frame.
    instrument_frame : str
        Instrument frame.

    Returns
    -------
    spacecraft_velocity : np.ndarray
        Particle velocity in the spacecraft frame.

    References
    ----------
    https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy
    """
    # Particle velocity in the pointing (DPS) frame wrt spacecraft.
    spacecraft_velocity = np.zeros((len(time), 3), dtype=np.float64)

    for index in range(len(time)):
        # Get and apply the rotation matrix to the particle velocity.
        spacecraft_velocity[index] = spice.mxv(
            spice.pxform(instrument_frame, "IMAP_DPS", time[index]),
            instrument_velocity[index],
        )

    return spacecraft_velocity
