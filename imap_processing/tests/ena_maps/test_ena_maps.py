"""Test classes and methods in ena_maps.py."""

from copy import deepcopy
from unittest import mock

import numpy as np
import pytest

from imap_processing.ena_maps import ena_maps
from imap_processing.spice import geometry
from imap_processing.tests.ultra.test_data.mock_data import mock_l1c_pset_product


@pytest.fixture()
def l1c_pset_products():
    """Make fake L1C Ultra PSET products for testing"""
    l1c_spatial_bin_spacing_deg = 10
    return {
        "spacing": l1c_spatial_bin_spacing_deg,
        "products": [
            mock_l1c_pset_product(
                spacing_deg=l1c_spatial_bin_spacing_deg,
                stripe_center_lon=mid_longitude,
                timestr=f"2025-09-{i + 1:02d}T12:00:00",
                head=("45" if (i % 2 == 0) else "90"),
            )
            for i, mid_longitude in enumerate(
                np.arange(
                    0,
                    360,
                    45,
                )
            )
        ],
    }


class TestUltraPointingSet:
    @pytest.fixture(autouse=True)
    def _setup_ultra_l1c_pset_products(self, l1c_pset_products):
        """Setup fixture data as class attributes"""
        self.l1c_spatial_bin_spacing_deg = l1c_pset_products["spacing"]
        self.l1c_pset_products = l1c_pset_products["products"]

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
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

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
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


class TestRectangularMap:
    @pytest.fixture(autouse=True)
    def _setup_ultra_l1c_pset_products(self, l1c_pset_products):
        """Setup fixture data as class attributes"""
        self.l1c_spatial_bin_spacing_deg = l1c_pset_products["spacing"]
        self.l1c_pset_products = l1c_pset_products["products"]
        self.pset_order = "C"
        self.ultra_psets = [
            ena_maps.UltraPointingSet(
                reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
                order=self.pset_order,
            )
            for l1c_product in self.l1c_pset_products
        ]

    def test_instantiate(self):
        """Test instantiation of RectangularMap"""
        rm = ena_maps.RectangularMap(
            spacing_deg=2,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
            order=self.pset_order,
        )

        # Check that the map is empty
        assert rm.data_dict == {}

        # Check that the reference frame is correctly set
        assert rm.reference_frame == geometry.SpiceFrame.ECLIPJ2000

        # Check that the order is correctly set
        assert rm.order == self.pset_order

        # Check the number of points is (360/2) * (180/2)
        np.testing.assert_equal(rm.num_points, int(360 * 180 / 4))

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    @pytest.mark.parametrize("map_spacing_deg", [2, 5, 10])
    @pytest.mark.parametrize("ravel_order", ["C", "F"])
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_match_pset_coords_to_indices_push_method(
        self, mock_frame_transform_az_el, map_spacing_deg, ravel_order
    ):
        """
        Test matching PSET coordinates to RectangularMap indices using "push" method.

        Parameterize by map_spacing_deg and ravel_order.
        """

        # Mock frame_transform to return the az and el, shifted by +13 degrees for luck
        def rotate_az_el_slightly(az_el):
            az_el += np.deg2rad(13)
            # Wrap az to [0, 2*pi) and el to [-pi/2, pi/2) radians
            az_el[:, 0] = az_el[:, 0] % (2 * np.pi)
            az_el[:, 1] = ((az_el[:, 1] + (np.pi / 2)) % np.pi) - (np.pi / 2)
            return az_el

        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: rotate_az_el_slightly(
                az_el
            )
        )
        rectangular_map = ena_maps.RectangularMap(
            spacing_deg=map_spacing_deg,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
            order=ravel_order,
        )
        # Find the indices of the map that match the PSET's az and el coordinates
        matched_indices = rectangular_map.match_pset_coords_to_indices(
            self.ultra_psets[0], ena_maps.IndexMatchMethod.PUSH
        )

        # The found az and el points should be the same as the input az and el points
        # to within the spacing of the map
        matched_map_az_el = rectangular_map.az_el_points[matched_indices]
        rotated_pset_az_el = self.ultra_psets[0].az_el_points
        np.testing.assert_allclose(
            matched_map_az_el[:, 1],
            rotated_pset_az_el[:, 1],
            atol=np.deg2rad(map_spacing_deg / 2),
        )

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_project_pset_values_to_map_push_method(self, mock_frame_transform_az_el):
        """
        Test projection of PSET values to RectangularMap w "push" index matching method.

        If frame_transform_az_el is mocked to return the az and el unchanged, and the
        map has the same spacing as the PSETs, then the map should have the same values
        as the PSETs, summed.
        """
        index_matching_method = ena_maps.IndexMatchMethod.PUSH

        pset_ravel_order = self.pset_order
        pset_spacing_deg = self.ultra_psets[0].spacing_deg

        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )

        rectangular_map = ena_maps.RectangularMap(
            spacing_deg=pset_spacing_deg,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
            order=pset_ravel_order,
        )

        # Project each PSET's values to the map
        for ultra_pset in self.ultra_psets:
            rectangular_map.project_pset_values_to_map(
                ultra_pset,
                value_keys=[
                    ("counts", index_matching_method),
                    ("exposure_time", index_matching_method),
                ],
            )

        # Check that the map has been updated
        assert rectangular_map.data_dict != {}

        # Check that the map has the same values as the PSETs, summed
        simple_summed_pset_counts = np.sum(
            [pset["counts"].values for pset in self.l1c_pset_products], axis=0
        ).reshape(rectangular_map.data_dict["counts"].shape, order=pset_ravel_order)

        np.testing.assert_allclose(
            rectangular_map.data_dict["counts"],
            simple_summed_pset_counts,
        )
