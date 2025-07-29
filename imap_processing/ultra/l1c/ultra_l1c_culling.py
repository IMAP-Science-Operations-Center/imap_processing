"""Culling for ULTRA L1c."""

import numpy as np
import astropy_healpix.healpy as hp
from numpy.typing import NDArray


from imap_processing.spice.geometry import (
    SpiceFrame,
    SpiceBody,
    imap_state,
)


def compute_culling_mask(
    et: NDArray,
    keepout_radius_km: float,
    observer=SpiceBody.EARTH,
    nside: int = 128,
    nested: bool = False,
) -> NDArray:
    """
    Compute a boolean mask for HEALPix pixels that are within a keep-out radius
    of the target body (e.g., Earth) in the spacecraft pointing frame.

    Parameters
    ----------
    et : np.ndarray
        Ephemeris times in TDB seconds past J2000.
    keepout_radius_km : float
        Radius (in km) within which HEALPix pixels will be excluded.
    observer : SpiceBody, optional
        Body from which IMAP is observed.
    nside : int, optional
        HEALPix NSIDE resolution. Default is 128.
    nested : bool, optional
        Whether to use NESTED indexing (default is RING).

    Returns
    -------
    mask : np.ndarray
        Boolean array of shape (len(et), npix).
    """

    # Compute number of HEALPix pixels
    npix = hp.nside2npix(nside)

    # Compute IMAP to Earth position in the pointing frame.
    state = imap_state(et, ref_frame=SpiceFrame.IMAP_DPS, observer=observer)
    position = -state[:, :3]  # Flip to get vector from IMAP to Earth

    # Distance from IMAP to target (e.g. Earth) (km):
    distance = np.linalg.norm(position, axis=1)  # shape (len(et),)

    # Calculate the keepout angle (radians).
    keepout_angle = np.arcsin(keepout_radius_km / distance)  # radians

    # Calculate the direction from IMAP to Earth. (shape: [N, 3])
    unit_target_vecs = position / distance[:, np.newaxis]

    # Calculate the direction of the HEALPix pixels. (shape: [npix, 3])
    pixel_vecs = hp.pix2vec(nside, np.arange(npix), nest=nested)
    pixel_vecs = np.vstack(pixel_vecs).T

    # Calculate distance from pixel to Earth.
    cos_sep = np.dot(unit_target_vecs, pixel_vecs.T)  # shape (N, npix)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    # Angular separation in radians
    sep_angle = np.arccos(cos_sep)  # shape (N, npix)

    # Exclude pixels within the keepout angle.
    mask = sep_angle > keepout_angle[:, np.newaxis]

    return mask
