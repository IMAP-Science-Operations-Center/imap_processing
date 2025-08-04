"""Tests Culling for ULTRA L1c."""

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import spiceypy

from imap_processing.ultra.l1c.ultra_l1c_culling import compute_culling_mask
from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
    imap_state,
)

@pytest.mark.external_kernel
@pytest.mark.usefixtures("_unset_metakernel_path")
def test_compute_culling_mask(furnish_kernels, spice_test_data_path):
    """Tests compute_culling_mask function."""

    planet_radii_km = {
        "EARTH": 6378.137,
    }

    kernels = [
        "imap_science_100.tf",
        "sim_1yr_imap_pointing_frame.bc",
        "imap_spk_demo.bsp",
    ]

    keepout_radius_km = 30 * planet_radii_km["EARTH"]

    # Corresponds to 2025-11-28T00:00:00
    et_start = 817561854.185627
    et_end = 817644684.1856259
    step_seconds = 1800  # 30 minutes
    et_steps = np.arange(et_start, et_end, step_seconds)

    spiceypy.kclear()

    with furnish_kernels(kernels):
        mask, _ = compute_culling_mask(et_steps, keepout_radius_km)

    assert mask.shape[0] == len(et_steps)
    assert mask.shape[1] == hp.nside2npix(128)

    # Check that some pixels are masked out
    assert not np.all(mask)
    assert np.any(mask)


@pytest.mark.external_kernel
@pytest.mark.usefixtures("_unset_metakernel_path")
def test_compare_sincpt_with_culling_mask_deterministic(furnish_kernels):
    """Systematically compare sincpt results to culling mask output at selected pixels."""

    with furnish_kernels([
        "imap_science_100.tf",
        "imap_sclk_0000.tsc",
        "sim_1yr_imap_pointing_frame.bc",
        "imap_spk_demo.bsp",
        "earth_1962_240827_2124_combined.bpc",
        "pck00011.tpc",
        "naif0012.tls",
        "de440s.bsp",
    ]):
        et = np.array([817561854.185627])
        keepout_radius_km = 6378.1  # Earth radius
        nside = 128
        npix = hp.nside2npix(nside)

        # Compute culling mask (True = KEEP, False = CULL)
        mask, unit_vectors = compute_culling_mask(et, keepout_radius_km, observer=SpiceBody.EARTH, nside=nside)

        # Get direction to Earth in IMAP_DPS frame (from IMAP to Earth)
        state = spiceypy.spkezr("EARTH", et[0], "IMAP_DPS", "NONE", "IMAP")[0]
        earth_dir = state[:3] / np.linalg.norm(state[:3])  # shape (3,)

        # Get pixel unit vectors in IMAP_DPS frame
        pixel_vecs_dps = np.column_stack(hp.pix2vec(nside, np.arange(npix)))  # shape (npix, 3)

        # Compute angular separation between pixel direction and Earth direction
        dot = np.dot(pixel_vecs_dps, earth_dir)
        dot = np.clip(dot, -1.0, 1.0)
        angles = np.arccos(dot)  # radians

        # Sort by angular separation from Earth direction
        sorted_indices = np.argsort(angles)

        # Convert pixel vectors to J2000 frame for sincpt
        rot_dps_to_j2000 = spiceypy.pxform("IMAP_DPS", "J2000", et[0])
        # Use index closes to the Earth
        pixel_vec_j2000 = np.dot(rot_dps_to_j2000, pixel_vecs_dps[sorted_indices[0]])

        masked = not mask[0, sorted_indices[0]]  # True if pixel is in keepout region
        hit = True  # default assumption

        try:
            spiceypy.sincpt(
                method="ELLIPSOID",
                target="EARTH",
                et=et[0],
                fixref="IAU_EARTH",
                abcorr="NONE",
                obsrvr="IMAP",
                dref="J2000",
                dvec=pixel_vec_j2000,
            )
        except spiceypy.utils.exceptions.NotFoundError:
            hit = False

        assert masked == hit
