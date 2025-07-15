"""Spacecraft state."""

from numpy.typing import NDArray

from imap_processing.spice.geometry import (
    SpiceFrame,
    imap_state,
)
from imap_processing.spice.time import met_to_sclkticks, sct_to_et


def calculate_gsm_state(met: NDArray) -> tuple[NDArray, NDArray]:
    """
    Calculate the position and velocity of the spacecraft in GSM coordinates.

    Parameters
    ----------
    met : NDArray
        Start time in UTC.

    Returns
    -------
    gsm_position: NDArray
        Spacecraft position in GSM Coordinates.
    gsm_velocity: NDArray
        Spacecraft velocity in GSM Coordinates.
    """
    sclk_ticks = met_to_sclkticks(met)
    et = sct_to_et(sclk_ticks)

    gsm_state = imap_state(et, ref_frame=SpiceFrame.IMAP_GSM)

    gsm_position = gsm_state[:, :3]
    gsm_velocity = gsm_state[:, 3:]

    return gsm_position, gsm_velocity
