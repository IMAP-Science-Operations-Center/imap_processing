"""Test classes and methods in ena_maps.py."""

from copy import deepcopy
from unittest import mock

import numpy as np
import pytest

from imap_processing.ena_maps import ena_maps
from imap_processing.spice import geometry
from imap_processing.tests.ultra.test_data.mock_data import mock_l1c_pset_product


class TestUltraPointingSet:
    @pytest.fixture()
    def _setup_l1c_pset_products(self):
        """Make fake L1C Ultra PSET products for testing"""
        self.l1c_spatial_bin_spacing_deg = 0.5
        self.l1c_pset_products = [
            mock_l1c_pset_product(
                num_lat_bins=int(180 / self.l1c_spatial_bin_spacing_deg),
                num_lon_bins=int(360 / self.l1c_spatial_bin_spacing_deg),
                stripe_center_lon_bin=mid_longitude,
                timestr=f"2025-09-{i + 1:02d}T12:00:00",
                head=("45" if (i % 2 == 0) else "90"),
            )
            for i, mid_longitude in enumerate(
                np.arange(
                    0,
                    int(360 / self.l1c_spatial_bin_spacing_deg),
                    int(45 / self.l1c_spatial_bin_spacing_deg),
                )
            )
        ]

    @pytest.mark.usefixtures("_setup_l1c_pset_products")
    def test_instantiate(self):
        """Test instantiation of UltraPointingSet"""
        ultra_psets = [
            ena_maps.UltraPointingSet(
                reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
                order="C",
            )
            for l1c_product in self.l1c_pset_products
        ]

        for ultra_pset in ultra_psets:
            # Check tiling is rectangular
            assert ultra_pset.tiling_type == ena_maps.TilingType.RECTANGULAR

            # Check that the reference frame is correctly set
            assert ultra_pset.reference_frame == geometry.SpiceFrame.IMAP_DPS

            # Check the number of points is (360/0.5) * (180/0.5)
            np.testing.assert_equal(
                ultra_pset.num_points,
                int(360 * 180 / (self.l1c_spatial_bin_spacing_deg**2)),
            )

    @pytest.mark.usefixtures("_setup_l1c_pset_products")
    @mock.patch("imap_processing.spice.geometry.frame_transform")
    def test_project_to_frame(self, mock_frame_transform):
        """Test projection of UltraPointingSet to a new frame"""

        # Mock frame_transform to return the negative of the input position vectors
        mock_frame_transform.side_effect = lambda et, pos, from_frame, to_frame: -pos

        ultra_psets = [
            ena_maps.UltraPointingSet(
                reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
                order="C",
            )
            for l1c_product in self.l1c_pset_products
        ]

        for ultra_pset in ultra_psets:
            original_pset = deepcopy(ultra_pset)

            # First projection inverts position vectors
            ultra_pset.project_to_frame(geometry.SpiceFrame.ECLIPJ2000)
            assert ultra_pset.reference_frame == geometry.SpiceFrame.ECLIPJ2000

            # Second projection inverts position vectors back to original
            # (check equal to original)
            ultra_pset.project_to_frame(geometry.SpiceFrame.IMAP_ULTRA_90)
            assert ultra_pset.reference_frame == geometry.SpiceFrame.IMAP_ULTRA_90
            np.testing.assert_allclose(
                ultra_pset.az_el_points, original_pset.az_el_points
            )

            # Third projection inverts position vectors again
            # (check not equal to original)
            ultra_pset.project_to_frame(geometry.SpiceFrame.J2000)
            assert ultra_pset.reference_frame == geometry.SpiceFrame.J2000
            assert not np.allclose(ultra_pset.az_el_points, original_pset.az_el_points)

            # Check that the history attribute has been updated (current frame last)
            assert ultra_pset.reference_frame_history == [
                geometry.SpiceFrame.IMAP_DPS,
                geometry.SpiceFrame.ECLIPJ2000,
                geometry.SpiceFrame.IMAP_ULTRA_90,
                geometry.SpiceFrame.J2000,
            ]
