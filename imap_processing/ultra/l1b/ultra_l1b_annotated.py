"""Calculates Annotated Events for ULTRA L1b."""

import typing

import numpy as np

from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform,
    imap_state,
)
from imap_processing.spice.kernels import ensure_spice


@ensure_spice
@typing.no_type_check
def get_particle_velocity(
    time: np.ndarray,
    instrument_velocity: np.ndarray,
    instrument_frame: SpiceFrame,
    pointing_frame: SpiceFrame,
) -> tuple[np.ndarray, np.ndarray]:
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
    particle_velocity_spacecraft : np.ndarray
        Particle velocity in the spacecraft frame.
    particle_velocity_heliosphere : np.ndarray
        Particle velocity in the heliosphere frame.

    References
    ----------
    https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy
    """
    # Particle velocity in the pointing (DPS) frame wrt spacecraft.
    particle_velocity_spacecraft = frame_transform(
        et=time,
        position=instrument_velocity,
        from_frame=instrument_frame,
        to_frame=pointing_frame,
    )

    # Spacecraft velocity in the pointing (DPS) frame wrt heliosphere.
    state = imap_state(time, ref_frame=SpiceFrame.IMAP_DPS)

    # Extract the velocity part of the state vector
    spacecraft_velocity = state[:, 3:6]

    # Apply Compton-Getting.
    # Particle velocity in the DPS frame wrt to the heliosphere
    particle_velocity_heliosphere = spacecraft_velocity + particle_velocity_spacecraft

    return particle_velocity_spacecraft, particle_velocity_heliosphere
