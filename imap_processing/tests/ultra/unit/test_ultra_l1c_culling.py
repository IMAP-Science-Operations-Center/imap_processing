"""Tests Culling for ULTRA L1c."""

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import spiceypy

from imap_processing.quality_flags import ImapPSETUltraFlags
from imap_processing.spice.geometry import SpiceBody
from imap_processing.ultra.l1c.ultra_l1c_culling import compute_culling_mask


@pytest.mark.external_kernel
def test_compute_culling_mask(furnish_kernels, spice_test_data_path):
    """Tests compute_culling_mask function."""

    planet_radii_km = {
        "EARTH": 6378.137,
    }

    kernels = [
        "imap_science_100.tf",
        "sim_1yr_imap_pointing_frame.bc",
        "imap_spk_demo.bsp",
        "imap_sclk_0000.tsc",
        "naif0012.tls",
    ]

    keepout_radius_km = 30 * planet_radii_km["EARTH"]

    # Corresponds to 2025-11-28T00:00:00
    et_start = 817561854.185627
    et_end = 817644684.1856259
    step_seconds = 1800  # 30 minutes
    et_steps = np.arange(et_start, et_end, step_seconds)
    nside = 128
    npix = hp.nside2npix(nside)

    spacecraft_pset_quality_flags = np.full(
        npix, ImapPSETUltraFlags.NONE.value, dtype=np.uint16
    )

    with furnish_kernels(kernels):
        mask, _ = compute_culling_mask(
            et_steps, keepout_radius_km, spacecraft_pset_quality_flags
        )

    assert mask.shape[0] == len(et_steps)
    assert mask.shape[1] == hp.nside2npix(128)

    # Check that some pixels are masked out
    assert not np.all(mask)
    assert np.any(mask)


@pytest.mark.external_kernel
def test_compare_sincpt_with_culling_mask_deterministic(furnish_kernels):
    """Compare culling mask output for the closest-to-Earth pixel with sincpt."""

    with furnish_kernels(
        [
            "imap_science_100.tf",
            "imap_sclk_0000.tsc",
            "sim_1yr_imap_pointing_frame.bc",
            "imap_spk_demo.bsp",
            "earth_latest_high_prec.bpc",
            "pck00011.tpc",
            "naif0012.tls",
            "de440s.bsp",
        ]
    ):
        et = np.array([817561854.185627, 817561854.185628])
        keepout_radius_km = 6378.1  # Earth radius
        nside = 128
        npix = hp.nside2npix(nside)
        spacecraft_pset_quality_flags = np.full(
            npix, ImapPSETUltraFlags.NONE.value, dtype=np.uint16
        )

        # Compute culling mask and IMAP-to-Earth unit vector
        mask, unit_vectors = compute_culling_mask(
            et,
            keepout_radius_km,
            spacecraft_pset_quality_flags,
            observer=SpiceBody.EARTH,
            nside=nside,
        )

        culled = np.any(~mask, axis=0)
        # Culled pixels must have the flag set (bitwise check is safest)
        assert np.all(
            (spacecraft_pset_quality_flags[culled] & ImapPSETUltraFlags.EARTH_FOV.value)
            == ImapPSETUltraFlags.EARTH_FOV.value
        )

        # Non-culled pixels must not have the flag
        assert np.all(
            (
                spacecraft_pset_quality_flags[~culled]
                & ImapPSETUltraFlags.EARTH_FOV.value
            )
            == 0
        )

        # Computes the 3D unit vectors pointing to the centers of all HEALPix pixels
        pixel_vecs_dps = np.column_stack(
            hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=False)
        )
        # Find the HEALPix pixel direction closest to the direction from IMAP to Earth
        closest_idx = np.argmax(np.dot(pixel_vecs_dps, unit_vectors[0]))

        # Transform closest pixel vector to J2000
        rot_dps_to_j2000 = spiceypy.pxform("IMAP_DPS", "J2000", et[0])
        pixel_vec_j2000 = np.dot(rot_dps_to_j2000, pixel_vecs_dps[closest_idx])

        with spiceypy.no_found_check():
            result = spiceypy.sincpt(
                method="ELLIPSOID",
                target="EARTH",
                et=et[0],
                fixref="IAU_EARTH",
                abcorr="NONE",
                obsrvr="IMAP",
                dref="J2000",
                dvec=pixel_vec_j2000,
            )

        assert result[-1]  # Check if the ray intersects Earth
