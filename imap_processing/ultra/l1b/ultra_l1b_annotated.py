"""Calculates Annotated Events for ULTRA L1b."""

import numpy as np
import spiceypy as spice
from numpy.typing import NDArray

from imap_processing.spice.kernels import ensure_spice


@ensure_spice
def get_particle_velocity(
    time: float,
    ultra_velocity: np.ndarray,
) -> NDArray:
    """
    Get the particle velocity in the pointing (DPS) frame wrt the spacecraft.

    Parameters
    ----------
    time : float
        Time in met.
    ultra_velocity : np.ndarray
        Particle velocity in the instrument frame.

    Returns
    -------
    spacecraft_velocity : np.ndarray
        Particle velocity in the spacecraft frame.
    """
    # Particle velocity in the pointing (DPS) frame wrt the spacecraft.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.mxv
    # https://spiceypy.readthedocs.io/en/main/
    # documentation.html#spiceypy.spiceypy.pxform

    #spacecraft_velocity = spice.mxv(
    #    spice.pxform("IMAP_SPACECRAFT", "IMAP_DPS", time), ultra_velocity
    #)
    spacecraft_velocities = spice.mxv(
            spice.pxform("IMAP_BODY", "IMAP_DPS", time), ultra_velocity
    )

    return spacecraft_velocities
