"""Calculates Annotated Events for ULTRA L1b."""

import typing

import numpy as np
import spiceypy as spice
from numpy.typing import NDArray

from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.kernels import ensure_spice
from imap_processing.spice.geometry import frame_transform


@ensure_spice
@typing.no_type_check
def get_particle_velocity(
    time: np.ndarray,
    instrument_velocity: np.ndarray,
    instrument_frame: SpiceFrame,
    pointing_frame: SpiceFrame,
) -> NDArray[np.float64]:
    """
    Get the particle velocity in the pointing (DPS) frame wrt the spacecraft.

    Parameters
    ----------
    time : np.ndarray
        Ephemeris time.
    instrument_velocity : np.ndarray
        Particle velocity in the instrument frame.
    instrument_frame : SpiceFrame
        Instrument frame.
    pointing_frame : SpiceFrame
        Pointing frame.

    Returns
    -------
    spacecraft_velocity : np.ndarray
        Particle velocity in the spacecraft frame.

    References
    ----------
    https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy
    """
    # Particle velocity in the pointing (DPS) frame wrt spacecraft.
    particle_velocity_spacecraft = frame_transform(
        et=time,
        position=instrument_velocity,
        from_frame=instrument_frame,
        to_frame=pointing_frame
    )

    # Spacecraft velocity in the pointing (DPS) frame wrt heliosphere.
    state, lt = spice.spkezr("IMAP", time, "IMAP_DPS", "NONE", "SUN")

    # Extract the velocity part of the state vector
    spacecraft_velocity = state[0][3:6]

    # Particle velocity in the DPS frame wrt to the heliosphere
    particle_velocity_heliosphere = spacecraft_velocity + \
                                    particle_velocity_spacecraft

    return particle_velocity_spacecraft, particle_velocity_heliosphere
