"""
Annotated Events: project events in instrument frame to reference systems.

References
----------
https://spiceypy.readthedocs.io/en/main/documentation.html.
"""

import logging

import numpy.typing as npt
import spiceypy

# Logger setup
logger = logging.getLogger(__name__)


def get_attitude_timerange(ck_kernel: str, id: int) -> tuple:
    """
    Get attitude timerange using the ck kernel.

    Parameters
    ----------
    ck_kernel : str
        Directory of the ck kernel.
    id : int
        ID of the instrument or spacecraft body.

    Returns
    -------
    start : float
        Start time in ET.
    end : float
        End time in ET.

    Notes
    -----
    ck_kernel refers to the kernel containing attitude data.
    """
    # Get the first CK path
    ck_path = ck_kernel[0]

    # Get the coverage window
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.ckcov
    cover = spiceypy.ckcov(ck_path, int(id), True, "SEGMENT", 0, "TDB")

    # TODO: Deal with gaps in coverage. For now, we use one interval.
    # How we interpolate may change when using type 2 kernels.
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.wncard
    spiceypy.wncard(cover)

    # Retrieve the start and end times of the coverage interval
    interval = 0
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.wnfetd
    start, end = spiceypy.wnfetd(cover, interval)

    # Convert start and end times from ET to UTC for human readability
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.et2utc
    start_utc = spiceypy.et2utc(start, "C", 0)
    end_utc = spiceypy.et2utc(end, "C", 0)

    logger.info(f"Coverage Interval for ID {id}:")
    logger.info(f"Start Time (UTC): {start_utc}")
    logger.info(f"End Time (UTC): {end_utc}")

    return start, end


def _get_particle_velocity(
    direct_events: dict,
) -> npt.NDArray:
    """
    Get the particle velocity in the heliosphere frame.

    Parameters
    ----------
    direct_events : dict
        Dictionary of direct events.

    Returns
    -------
    vultra_heliosphere_frame : numpy.ndarray
        Particle velocity in the heliosphere frame.

    Notes
    -----
    Using a single event time/velocity for now.
    This of course will change.
    """
    time = direct_events["tdb"]
    ultra_velocity = direct_events["vultra"]  # particle velocity in the ultra frame

    # Particle velocity from instrument to dps frame
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.mxv
    # https://spiceypy.readthedocs.io/en/main/
    # documentation.html#spiceypy.spiceypy.pxform
    dps_velocity = spiceypy.mxv(
        spiceypy.pxform("IMAP_ULTRA_45", "IMAP_DPS", time), ultra_velocity
    )

    # Spacecraft velocity in the DPS frame wrt the heliosphere
    # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.spkezr
    state, lt = spiceypy.spkezr("IMAP", time, "IMAP_DPS", "NONE", "SUN")

    # Extract the velocity part of the state vector
    imap_dps_velocity = state[3:6]

    # Particle velocity in the DPS frame wrt to the heliosphere
    ultra_velocity_heliosphere_frame = imap_dps_velocity + dps_velocity

    return ultra_velocity_heliosphere_frame


def build_annotated_events(direct_events: dict, kernels: list) -> None:
    """
    Build annotated events.

    Parameters
    ----------
    direct_events : dict
        Dictionary of direct events.
    kernels : list
        List of kernels to be loaded.
    """
    with spiceypy.KernelPool(kernels):
        # Hack: create time for IMAP_DPS frame
        # dps = despun reference frame.
        # TODO: compute the mean z-axis for a given pointing and
        # put it in the center of that pointing (time)
        # which will be the spacecraft z-axis (in eclipj2000)
        # and then use this to create a single DPS frame for each pointing
        # In other words, this will dynamically change for each pointing.
        # We need a CK that defines the DPS frame. Nick Dutton putting
        # together a memo on how this will work.

        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.pdpool
        spiceypy.pdpool("FRAME_-43906_FREEZE_EPOCH", [802094471.0])

        # https://spiceypy.readthedocs.io/en/main/documentation.html#spiceypy.spiceypy.et2utc
        time_event = spiceypy.et2utc(direct_events["tdb"], "C", 0)

        logger.info(f"Time (UTC): {time_event}")

        _get_particle_velocity(direct_events)
