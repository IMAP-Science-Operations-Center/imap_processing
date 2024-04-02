"""Annotated Events: project events in instrument frame to reference systems.

Reference: https://spiceypy.readthedocs.io/en/main/documentation.html.
"""

import logging

import spiceypy as spice

from tools.spice.spice_utils import list_all_constants

# Logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_attitude_timerange(ck_kernel, id):
    """Get attitude timerange using the ck kernel.

    Parameters
    ----------
    ck_kernel: str
        Directory of the ck kernel.
    id: int
        ID of the instrument or spacecraft body.

    Returns
    -------
    start: float
        Start time in ET.
    end: float
        End time in ET.
    """
    # Get the first CK path
    ck_path = ck_kernel[0]

    # Get the IDs from the CK file
    ck_id = spice.ckobj(ck_kernel[0])

    # Check if the specified ID is in the CK file
    if int(id) not in ck_id:
        raise ValueError(f"ID {id} not found in CK file")

    # Get the coverage window
    cover = spice.ckcov(ck_path, int(id), True, "SEGMENT", 0, "TDB")

    # TODO: Deal with gaps in coverage. For now, we use one interval.
    # How we interpolate may change when using type 2 kernels.
    spice.wncard(cover)

    # Retrieve the start and end times of the coverage interval
    interval = 0
    start, end = spice.wnfetd(cover, interval)

    # Convert start and end times from ET to UTC for human readability
    start_utc = spice.et2utc(start, "C", 0)
    end_utc = spice.et2utc(end, "C", 0)

    logger.info(f"Coverage Interval for ID {id}:")
    logger.info(f"Start Time (UTC): {start_utc}")
    logger.info(f"End Time (UTC): {end_utc}")

    return start, end


def get_particle_velocity(direct_events):
    """Get the particle velocity in the heliosphere frame.

    Parameters
    ----------
    direct_events: dict
        Dictionary of direct events.

    Returns
    -------
    vultra_heliosphere_frame: np.ndarray
        Particle velocity in the heliosphere frame.

    #Note: Using a single event time/velocity for now.
    This of course will change.
    """
    time = direct_events["tdb"]
    vultra = direct_events["vultra"]  # particle velocity in the ultra frame

    # Particle velocity from instrument to spacecraft frame
    vscbody = spice.mxv(spice.pxform("IMAP_ULTRA_45", "IMAP_SPACECRAFT", time), vultra)

    # Particle velocity from spacecraft to dps frame
    vdpssc = spice.mxv(spice.pxform("IMAP_SPACECRAFT", "IMAP_DPS", time), vscbody)

    # Spacecraft velocity in the DPS frame wrt the heliosphere
    state, lt = spice.spkezr("IMAP", time, "IMAP_DPS", "NONE", "SUN")

    # Extract the velocity part of the state vector
    imapdpsvel = state[3:6]

    # Particle velocity in the DPS frame wrt to the heliosphere
    vultra_heliosphere_frame = imapdpsvel + vdpssc

    return vultra_heliosphere_frame


def build_annotated_events(direct_events, kernels):
    """Build annotated events.

    Parameters
    ----------
    direct_events: dict
        Dictionary of direct events.
    kernels: list
        List of kernels to be loaded.
    """
    with spice.KernelPool(kernels):
        # Hack: create time for IMAP_DPS frame
        # dps = despun reference frame.
        # TODO: compute the mean z-axis for a given pointing and
        # put it in the center of that pointing (time)
        # which will be the spacecraft z-axis (in eclipj2000)
        # and then use this to create a single DPS frame for each pointing
        # In other words, this will dynamically change for each pointing.
        # We need a CK that defines the DPS frame. Nick Dutton putting
        # together a memo on how this will work.
        spice.pdpool("FRAME_-43906_FREEZE_EPOCH", [802094471.0])

        time_event = spice.et2utc(direct_events["tdb"], "C", 0)

        logger.info(f"Time (UTC): {time_event}")

        # Just for demo purposes
        list_all_constants()

        get_particle_velocity(direct_events)
