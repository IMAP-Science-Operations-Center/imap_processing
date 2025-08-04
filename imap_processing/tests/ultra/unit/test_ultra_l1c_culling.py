"""Tests Culling for ULTRA L1c."""

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import spiceypy

from imap_processing.ultra.l1c.ultra_l1c_culling import compute_culling_mask


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
        mask = compute_culling_mask(et_steps, keepout_radius_km)

    assert mask.shape[0] == len(et_steps)
    assert mask.shape[1] == hp.nside2npix(128)

    # Check that some pixels are masked out
    assert not np.all(mask)
    assert np.any(mask)


import pytest


@pytest.mark.external_kernel
@pytest.mark.usefixtures("_unset_metakernel_path")
def test_sincpt_hits_and_misses(furnish_kernels):
    """Test that sincpt finds an Earth intercept when pointed directly at Earth,
    and fails when slightly off-axis."""

    with furnish_kernels(
        [
            "imap_science_100.tf",
            "imap_sclk_0000.tsc",
            "sim_1yr_imap_pointing_frame.bc",
            "imap_spk_demo.bsp",
            "earth_1962_240827_2124_combined.bpc",
            "pck00011.tpc",
            "naif0012.tls",
            "de440s.bsp",
        ]
    ):
        et = 817561854.185627

        # Earth direction from IMAP in J2000
        state, _ = spiceypy.spkezr("EARTH", et, "J2000", "NONE", "IMAP")
        earth_vec = state[:3] / np.linalg.norm(state[:3])

        # Direct hit
        try:
            spiceypy.sincpt(
                method="ELLIPSOID",
                target="EARTH",
                et=et,
                fixref="IAU_EARTH",
                abcorr="NONE",
                obsrvr="IMAP",
                dref="J2000",
                dvec=earth_vec,
            )
            hit = True
        except spiceypy.utils.exceptions.NotFoundError:
            hit = False

        assert hit, "Expected sincpt to return a hit for direct Earth vector"

        # Off-axis miss (~1 degree away)
        off_axis_vec = spiceypy.vrotv(earth_vec, [0, 1, 0], np.radians(1.0))
        try:
            spiceypy.sincpt(
                method="ELLIPSOID",
                target="EARTH",
                et=et,
                fixref="IAU_EARTH",
                abcorr="NONE",
                obsrvr="IMAP",
                dref="J2000",
                dvec=off_axis_vec,
            )
            hit_off = True
        except spiceypy.utils.exceptions.NotFoundError:
            hit_off = False

        assert not hit_off, "Expected sincpt to miss Earth when vector is 1° off-axis"
