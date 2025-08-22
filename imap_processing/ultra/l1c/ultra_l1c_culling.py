"""Culling for ULTRA L1c."""

import astropy_healpix.healpy as hp
import numpy as np
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapPSETUltraFlags
from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
    imap_state,
)


def compute_culling_mask(
    et: NDArray,
    keepout_radius_km: float,
    pset_quality_flags: NDArray,
    observer: SpiceBody = SpiceBody.EARTH,
    nside: int = 128,
    nested: bool = False,
) -> tuple[NDArray, NDArray]:
    """
    Compute a mask for HEALPix pixels within a keep-out radius of the target body.

    Parameters
    ----------
    et : NDArray
        Ephemeris times in TDB seconds past J2000.
    keepout_radius_km : float
        Radius (in km) within which HEALPix pixels will be excluded.
    pset_quality_flags : NDArray,
        Quality flag to set when HEALPIX pixels are within a
        keep-out radius of the target body.
    observer : SpiceBody, optional
        Body from which IMAP is observed.
    nside : int, optional
        HEALPix NSIDE resolution. Default is 128.
    nested : bool, optional
        Whether to use NESTED indexing.

    Returns
    -------
    mask : tuple[NDArray, NDArray]
        Boolean array of shape (len(et), npix).
    unit_target_vecs : NDArray
        Unit vectors from IMAP to the target body
        (e.g., Earth), shape (len(et), 3).
    """
    # Compute number of HEALPix pixels
    npix = hp.nside2npix(nside)

    # Compute IMAP to Earth position in the pointing frame.
    state = imap_state(et, ref_frame=SpiceFrame.IMAP_DPS, observer=observer)
    # Flip to get vector from IMAP to Earth
    # position.shape = (len(et), 3)
    position = -state[:, :3]

    # Distance from IMAP to target (e.g. Earth) (km):
    # distance.shape = (len(et),)
    distance = np.linalg.norm(position, axis=1)  # shape (len(et),)

    # Calculate the keepout angle (radians).
    # keepout_angle.shape = (len(et),)
    keepout_angle = np.arcsin(keepout_radius_km / distance)  # radians

    # Calculate the direction from IMAP to Earth. (shape: [N, 3])
    # unit_target_vecs.shape = (len(et), 3)
    unit_target_vecs = position / distance[:, np.newaxis]

    # Get pixel unit vectors pointing from the center of the
    # HEALPix sphere to the center of each pixel on the sky.
    pixel_vecs = np.column_stack(
        hp.pix2vec(nside, np.arange(npix), nest=nested)
    )  # shape: (npix, 3)

    # Returns cos(theta) where theta is the separation angle between:
    # (1) vector from IMAP to Earth
    # (2) vector from IMAP to HEALPix pixel center
    # If theta is within the keepout angle, then the pixel is culled.
    cos_sep = np.dot(unit_target_vecs, pixel_vecs.T)  # shape (N, npix)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    # Get theta here.
    sep_angle = np.arccos(cos_sep)

    # Exclude pixels within the keepout angle.
    # mask.shape = (len(et), npix)
    mask = sep_angle > keepout_angle[:, np.newaxis]
    culled_any_time = np.any(~mask, axis=0)  # shape: (npix,)
    pset_quality_flags[culled_any_time] |= ImapPSETUltraFlags.EARTH_FOV.value

    return mask, unit_target_vecs
