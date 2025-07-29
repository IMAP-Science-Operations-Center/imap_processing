"""Tests Culling for ULTRA L1c."""
import matplotlib
matplotlib.use("TkAgg")  # or try "QtAgg" if installed

import numpy as np
import matplotlib.pyplot as plt
import pytest
import spiceypy
import astropy_healpix.healpy as hp

from imap_processing.ultra.l1c.ultra_l1c_culling import (
    compute_culling_mask
)
from imap_processing.spice.time import str_to_et


@pytest.mark.external_kernel
@pytest.mark.usefixtures("_unset_metakernel_path")
def test_compute_culling_mask(furnish_kernels):
    """Tests compute_culling_mask function."""

    PLANET_RADII_KM = {
        "EARTH": 6378.137,
        "MOON": 1737.4,
        "JUPITER": 71492.0,
    }

    kernels = [
        "imap_science_100.tf",
        "imap_sclk_0000.tsc",
        "sim_1yr_imap_attitude.bc",
        "imap_wkcp.tf",
        "naif0012.tls",
        "sim_1yr_imap_pointing_frame.bc",
        "de440s.bsp",
        "imap_spk_demo.bsp",
    ]

    keepout_radius_km = 30 * PLANET_RADII_KM["EARTH"]

    et_start = str_to_et("2025-11-28T00:00:00")
    et_end = et_start + 24 * 60 * 60  # 24 hours
    step_seconds = 1800
    et_steps = np.arange(et_start, et_end + step_seconds, step_seconds)
    et_steps = np.array([797949054.185627])

    spiceypy.kclear()

    with furnish_kernels(kernels):
        mask = compute_culling_mask(et_steps, keepout_radius_km)
    import healpy as hp
    hp.mollview(
        mask[0].astype(float),  # Use the first (and only) time slice
        title="Culling Mask on 2025-11-28",
        coord="C",
        cmap="viridis",
        flip="astro",
    )

    plt.show()
    print('hi')



