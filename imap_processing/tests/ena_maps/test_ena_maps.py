"""Test classes and methods in ena_maps.py."""

from __future__ import annotations

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


# class TestUltraPointingSet:
#     @pytest.fixture(autouse=True)
#     def _setup_ultra_l1c_pset_products(self, l1c_pset_products):
#         """Setup fixture data as class attributes"""
#         self.l1c_spatial_bin_spacing_deg = l1c_pset_products["spacing"]
#         self.l1c_pset_products = l1c_pset_products["products"]

#     @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
#     def test_instantiate(self):
#         """Test instantiation of UltraPointingSet"""
#         ultra_psets = [
#             ena_maps.UltraPointingSet(
#                 pset_frame=geometry.SpiceFrame.IMAP_DPS,
#                 l1c_dataset=l1c_product,
#                 order="C",
#             )
#             for l1c_product in self.l1c_pset_products
#         ]

#         for ultra_pset in ultra_psets:
#             # Check tiling is rectangular
#             assert ultra_pset.tiling_type == ena_maps.SkyTilingType.RECTANGULAR

#             # Check that the reference frame is correctly set
#             assert ultra_pset.pset_frame == geometry.SpiceFrame.IMAP_DPS

#             # Check the number of points is (360/0.5) * (180/0.5)
#             np.testing.assert_equal(
#                 ultra_pset.num_points,
#                 int(360 * 180 / (self.l1c_spatial_bin_spacing_deg**2)),
#             )

#             # Check the repr exists
#             assert "UltraPointingSet" in repr(ultra_pset)

#     @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
#     @mock.patch("imap_processing.spice.geometry.frame_transform")
#     def test_project_to_frame(self, mock_frame_transform):
#         """Test projection of UltraPointingSet to a new frame"""

#         # Mock frame_transform to return the negative of the input position vectors
#         mock_frame_transform.side_effect = lambda et, pos, from_frame, to_frame: -pos

#         ultra_psets = [
#             ena_maps.UltraPointingSet(
#                 pset_frame=geometry.SpiceFrame.IMAP_DPS,
#                 l1c_dataset=l1c_product,
#                 order="C",
#             )
#             for l1c_product in self.l1c_pset_products
#         ]

#         for ultra_pset in ultra_psets:
#             original_pset = deepcopy(ultra_pset)

#             # First projection inverts position vectors
#             ultra_pset.project_to_frame(geometry.SpiceFrame.ECLIPJ2000)
#             assert ultra_pset.pset_frame == geometry.SpiceFrame.ECLIPJ2000

#             # Second projection inverts position vectors back to original
#             # (check equal to original)
#             ultra_pset.project_to_frame(geometry.SpiceFrame.IMAP_ULTRA_90)
#             assert ultra_pset.pset_frame == geometry.SpiceFrame.IMAP_ULTRA_90
#             np.testing.assert_allclose(
#                 ultra_pset.az_el_points, original_pset.az_el_points
#             )

#             # Third projection inverts position vectors again
#             # (check not equal to original)
#             ultra_pset.project_to_frame(geometry.SpiceFrame.J2000)
#             assert ultra_pset.pset_frame == geometry.SpiceFrame.J2000
# assert not np.allclose(
#     ultra_pset.az_el_points, original_pset.az_el_points)

#             # Check that the history attribute has been updated (current frame last)
#             assert ultra_pset.pset_frame_history == [
#                 geometry.SpiceFrame.IMAP_DPS,
#                 geometry.SpiceFrame.ECLIPJ2000,
#                 geometry.SpiceFrame.IMAP_ULTRA_90,
#                 geometry.SpiceFrame.J2000,
#             ]

#     @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
#     def test_uneven_spacing_raises_error(self):
#         """Test that uneven spacing in az/el raises ValueError"""

#         # Create dataset with uneven az spacing
#         uneven_az_dataset = xr.Dataset()
#         uneven_az_dataset["epoch"] = 1
#         uneven_az_dataset["azimuth_bin_center"] = np.array([0, 5, 15, 20, 30])
#         uneven_az_dataset["elevation_bin_center"] = np.arange(5)

#         with pytest.raises(ValueError, match="Azimuth bin spacing is not uniform"):
#             ena_maps.UltraPointingSet(
#                 pset_frame=geometry.SpiceFrame.IMAP_DPS,
#                 l1c_dataset=uneven_az_dataset,
#             )

#         uneven_az_dataset["azimuth_bin_center"] = np.arange(5)
#         uneven_az_dataset["elevation_bin_center"] = np.array([0, 5, 15, 20, 30])

#         with pytest.raises(ValueError, match="Elevation bin spacing is not uniform"):
#             ena_maps.UltraPointingSet(
#                 pset_frame=geometry.SpiceFrame.IMAP_DPS,
#                 l1c_dataset=uneven_az_dataset,
#             )

#         # Even but not the same spacing between az and el
#         uneven_az_dataset["azimuth_bin_center"] = np.arange(5)
#         uneven_az_dataset["elevation_bin_center"] = np.arange(5) * 2

#         with pytest.raises(
#             ValueError, match="Azimuth and elevation bin spacing do not match:"
#         ):
#             ena_maps.UltraPointingSet(
#                 pset_frame=geometry.SpiceFrame.IMAP_DPS,
#                 l1c_dataset=uneven_az_dataset,
#             )


# class TestRectangularSkyMap:
#     @pytest.fixture(autouse=True)
#     def _setup_ultra_l1c_pset_products(self, l1c_pset_products):
#         """Setup fixture data as class attributes"""
#         self.l1c_spatial_bin_spacing_deg = l1c_pset_products["spacing"]
#         self.l1c_pset_products = l1c_pset_products["products"]
#         self.pset_order = "C"
#         self.ultra_psets = [
#             ena_maps.UltraPointingSet(
#                 pset_frame=geometry.SpiceFrame.IMAP_DPS,
#                 l1c_dataset=l1c_product,
#                 order=self.pset_order,
#             )
#             for l1c_product in self.l1c_pset_products
#         ]

#     def test_instantiate(self):
#         """Test instantiation of RectangularSkyMap"""
#         rm = ena_maps.RectangularSkyMap(
#             spacing_deg=2,
#             spice_frame=geometry.SpiceFrame.ECLIPJ2000,
#             order=self.pset_order,
#         )

#         # Check that the map is empty
#         assert rm.data_dict == {}

#         # Check that the reference frame is correctly set
#         assert rm.reference_frame == geometry.SpiceFrame.ECLIPJ2000

#         # Check that the order is correctly set
#         assert rm.order == self.pset_order

#         # Check the number of points is (360/2) * (180/2)
#         np.testing.assert_equal(rm.num_points, int(360 * 180 / 4))

#         # Check the repr exists
#         assert "RectangularSkyMap" in repr(rm)

#     @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
#     @pytest.mark.parametrize("map_spacing_deg", [2, 5, 10])
#     @pytest.mark.parametrize("ravel_order", ["C", "F"])
#     @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
#     def test_match_pset_coords_to_indices_push_method(
#         self, mock_frame_transform_az_el, map_spacing_deg, ravel_order
#     ):
#         """
#         Test matching PSET coordinates to RectangularSkyMap
#         indices using "push" method.

#         Parameterize by map_spacing_deg and ravel_order.
#         """

#         # Mock frame_transform to return the az and el,
#         # shifted by +13 degrees for luck
#         def rotate_az_el_slightly(az_el):
#             az_el += np.deg2rad(13)
#             # Wrap az to [0, 2*pi) and el to [-pi/2, pi/2) radians
#             az_el[:, 0] = az_el[:, 0] % (2 * np.pi)
#             az_el[:, 1] = ((az_el[:, 1] + (np.pi / 2)) % np.pi) - (np.pi / 2)
#             return az_el

#         mock_frame_transform_az_el.side_effect = (
#             lambda et, az_el, from_frame, to_frame, degrees: rotate_az_el_slightly(
#                 az_el
#             )
#         )
#         rectangular_map = ena_maps.RectangularSkyMap(
#             spacing_deg=map_spacing_deg,
#             spice_frame=geometry.SpiceFrame.ECLIPJ2000,
#             order=ravel_order,
#         )
#         # Find the indices of the map that match the PSET's az and el coordinates
#         matched_indices = rectangular_map.match_pset_coords_to_indices(
#             self.ultra_psets[0], ena_maps.IndexMatchMethod.PUSH
#         )

#         # The found az and el points should be the same as the input az and el points
#         # to within the spacing of the map
#         matched_map_az_el = rectangular_map.az_el_points[matched_indices]
#         rotated_pset_az_el = self.ultra_psets[0].az_el_points
#         np.testing.assert_allclose(
#             matched_map_az_el[:, 1],
#             rotated_pset_az_el[:, 1],
#             atol=np.deg2rad(map_spacing_deg / 2),
#         )

#     @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
#     @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
#     def test_project_pset_values_to_map_push_method(self, mock_frame_transform_az_el):
#         """
#         Test projection of PSET values to Rect. Map w "push" index matching method.

#         If frame_transform_az_el is mocked to return the az and el unchanged, and the
#         map has the same spacing as the PSETs, then the map should have
#         the same values as the PSETs, summed.
#         """
#         index_matching_method = ena_maps.IndexMatchMethod.PUSH

#         pset_ravel_order = self.pset_order
#         pset_spacing_deg = self.ultra_psets[0].spacing_deg

#         # Mock frame_transform to return the az and el unchanged
#         mock_frame_transform_az_el.side_effect = (
#             lambda et, az_el, from_frame, to_frame, degrees: az_el
#         )

#         rectangular_map = ena_maps.RectangularSkyMap(
#             spacing_deg=pset_spacing_deg,
#             spice_frame=geometry.SpiceFrame.ECLIPJ2000,
#             order=pset_ravel_order,
#         )

#         # Project each PSET's values to the map
#         for ultra_pset in self.ultra_psets:
#             rectangular_map.project_pset_values_to_map(
#                 ultra_pset,
#                 value_keys=[
#                     ("counts", index_matching_method),
#                     ("exposure_time", index_matching_method),
#                 ],
#             )

#         # Check that the map has been updated
#         assert rectangular_map.data_dict != {}

#         # Check that the map has the same values as the PSETs, summed
#         simple_summed_pset_counts = np.sum(
#             [pset["counts"].values for pset in self.l1c_pset_products], axis=0
#         ).reshape(rectangular_map.data_dict["counts"].shape, order=pset_ravel_order)

#         np.testing.assert_allclose(
#             rectangular_map.data_dict["counts"],
#             simple_summed_pset_counts,
#         )

#     @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
#     @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
#     def test_project_pset_values_to_map_pull_method(self, mock_frame_transform_az_el):
#         """Test projection to Rect. Map fails w "pull" index matching method."""

#         index_matching_method = ena_maps.IndexMatchMethod.PULL
#         rectangular_map = ena_maps.RectangularSkyMap(
#             spacing_deg=10,
#             spice_frame=geometry.SpiceFrame.ECLIPJ2000,
#             order=self.pset_order,
#         )

#         with pytest.raises(NotImplementedError):
#             rectangular_map.project_pset_values_to_map(
#                 self.ultra_psets[0],
#                 value_keys=[
#                     ("counts", index_matching_method),
#                     ("exposure_time", index_matching_method),
#                 ],
#             )


class TestIndexMatching:
    @pytest.fixture(autouse=True)
    def _setup_ultra_l1c_pset_products(self, l1c_pset_products):
        """Setup fixture data as class attributes"""
        self.l1c_spatial_bin_spacing_deg = l1c_pset_products["spacing"]
        self.l1c_pset_products = l1c_pset_products["products"]

    @pytest.mark.parametrize(
        "map_spacing_deg",
        [0.5, 1, 10],
    )
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_match_coords_to_indices_pset_to_rect_map(
        self, mock_frame_transform_az_el, map_spacing_deg
    ):
        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )

        # Mock a PSET, overriding the az/el points
        mock_pset_input_frame = ena_maps.UltraPointingSet(
            l1c_dataset=self.l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        manual_az_el_coords = np.array(
            [
                [0, -90],  # always -> RectangularSkyMap pixel 0
                [0.4999999, -90],
                [180.5, -89.5],
                [359.5, -89.5],
                [0.5, 0],
                [180.5, 0],
                [359.5, 0],
                [0.5, 89.5],
                [180.5, 89.5],
                [359.5, 89.5],
                [359.999999, 89.99999],
            ]
        )
        mock_pset_input_frame.az_el_points = np.deg2rad(manual_az_el_coords)

        # Manually calculate the resulting 1D pixel indices for each az/el pair
        # (num of pixels in an az row spanning 180 deg of elevation) * (current az row)
        # + (pixel along in current az row)
        expected_output_pixel = np.array(
            [
                (az // map_spacing_deg) * (180 // map_spacing_deg)
                + ((90 + el) // map_spacing_deg)
                for [az, el] in manual_az_el_coords
            ]
        )

        # Mock the rectangular map and check the output values
        mock_rect_map = ena_maps.RectangularSkyMap(
            spacing_deg=map_spacing_deg,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        flat_indices_input_grid_output_frame = ena_maps.match_coords_to_indices(
            mock_pset_input_frame, mock_rect_map
        )
        assert mock_rect_map.num_points == 360 * 180 / map_spacing_deg**2
        assert len(flat_indices_input_grid_output_frame) == len(manual_az_el_coords)
        np.testing.assert_equal(
            flat_indices_input_grid_output_frame, expected_output_pixel
        )

    def test_match_coords_to_indices_pset_to_healpix_map_other_map(
        self,
    ):
        mock_pset_input_frame = ena_maps.UltraPointingSet(
            l1c_dataset=self.l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        # Until implemented, just change the tiling on a RectangularSkyMap
        mock_hp_map = ena_maps.RectangularSkyMap(
            spacing_deg=2,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        mock_hp_map.tiling_type = ena_maps.SkyTilingType.HEALPIX

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            ena_maps.match_coords_to_indices(mock_pset_input_frame, mock_hp_map)

        mock_other_map = mock_hp_map
        mock_other_map.tiling_type = "INVALID"
        with pytest.raises(ValueError, match="Tiling type of the output frame"):
            ena_maps.match_coords_to_indices(mock_pset_input_frame, mock_other_map)
