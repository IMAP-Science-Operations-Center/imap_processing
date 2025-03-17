"""Shared fixtures for ENA maps tests."""

import numpy as np
import pytest

from imap_processing.tests.ultra.test_data.mock_data import (
    mock_l1c_pset_product_healpix,
    mock_l1c_pset_product_rectangular,
)


@pytest.fixture(scope="module")
def ultra_l1c_pset_datasets():
    """Make fake L1C Ultra PSET products on a HEALPix tiling for testing"""
    l1c_nside = 64
    return {
        "nside": l1c_nside,
        "products": [
            mock_l1c_pset_product_healpix(
                nside=l1c_nside,
                stripe_center_lat=mid_latitude,
                width_scale=5,
                counts_scaling_params=(50, 0.5),
                peak_exposure=1000,
                timestr=f"2025-09-{i + 1:02d}T12:00:00",
                head="90",
            )
            for i, mid_latitude in enumerate(np.arange(-90, 90, 22.5))
        ],
    }


@pytest.fixture(scope="session")
def rectangular_l1c_pset_datasets():
    """Make fake L1C Ultra PSET products on a rectangular tiling for testing"""
    l1c_spacing_deg = 1
    return {
        "spacing": l1c_spacing_deg,
        "products": [
            mock_l1c_pset_product_rectangular(
                spacing_deg=l1c_spacing_deg,
                stripe_center_lat=mid_latitude,
                width_scale=5,
                counts_scaling_params=(50, 0.5),
                peak_exposure=1000,
                timestr=f"2025-09-{i + 1:02d}T12:00:00",
                head="90",
            )
            for i, mid_latitude in enumerate(np.arange(-90, 90, 22.5))
        ],
    }
