"""Tests Culling for ULTRA L1c."""

import matplotlib

matplotlib.use("TkAgg")  # or "QtAgg" if you have it

import astropy_healpix.healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import spiceypy
from scipy.interpolate import griddata

from imap_processing.ultra.l1c.ultra_l1c_culling import compute_culling_mask


@pytest.mark.external_kernel
@pytest.mark.usefixtures("_unset_metakernel_path")
def test_compute_culling_mask(furnish_kernels, spice_test_data_path):
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

    # Corresponds to 2025-11-28T00:00:00
    et_start = 817561854.185627
    et_end = 817644684.1856259  # 24 hours
    step_seconds = 1800
    et_steps = np.arange(et_start, et_end + step_seconds, step_seconds)
    et_steps = np.array([817561854.185627])

    spiceypy.kclear()
    with furnish_kernels(kernels):
        mask = compute_culling_mask(et_steps, keepout_radius_km)

    # HEALPix to RA/Dec
    mask_float = mask[0].astype(float)
    nside = hp.npix2nside(mask_float.size)
    pix_indices = np.arange(mask_float.size)
    vecs = np.vstack(hp.pix2vec(nside, pix_indices, nest=False)).T
    ra_deg, dec_deg = hp.vec2ang(vecs, lonlat=True)

    # RA/Dec interpolation grid (0.5 deg resolution)
    ra_vals = np.linspace(0, 360, 720)
    dec_vals = np.linspace(-90, 90, 360)
    ra_grid, dec_grid = np.meshgrid(ra_vals, dec_vals)

    # Interpolate the mask
    mask_grid = griddata(
        points=(ra_deg, dec_deg),
        values=mask_float,
        xi=(ra_grid, dec_grid),
        method="nearest",
        fill_value=np.nan,
    )

    # Plot as square image
    plt.figure(figsize=(12, 6))
    plt.imshow(
        mask_grid,
        origin="lower",
        extent=(0, 360, -90, 90),
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    plt.colorbar(label="Culling Mask Value (1=Visible, 0=Culled)")
    plt.title("Culling Mask on 2025-11-28 (RA/Dec Grid)")
    plt.xlabel("Right Ascension [deg]")
    plt.ylabel("Declination [deg]")
    plt.tight_layout()
    plt.show()
    print("hi")
