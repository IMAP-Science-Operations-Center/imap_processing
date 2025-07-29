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
