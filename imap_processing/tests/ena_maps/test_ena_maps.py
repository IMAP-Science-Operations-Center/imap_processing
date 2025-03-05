"""Test classes and methods in ena_maps.py."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing.ena_maps import ena_maps
from imap_processing.spice import geometry
from imap_processing.tests.ultra.test_data.mock_data import mock_l1c_pset_product


@pytest.fixture()
def l1c_pset_products():
    """Make fake L1C Ultra PSET products for testing"""
    l1c_spatial_bin_spacing_deg = 2
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
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
            )
            for l1c_product in self.l1c_pset_products
        ]

        for ultra_pset in ultra_psets:
            # Check tiling is rectangular
            assert ultra_pset.tiling_type == ena_maps.SkyTilingType.RECTANGULAR

            # Check that the reference frame is correctly set
            assert ultra_pset.spice_reference_frame == geometry.SpiceFrame.IMAP_DPS

            # Check the number of points is (360/0.5) * (180/0.5)
            np.testing.assert_equal(
                ultra_pset.num_points,
                int(360 * 180 / (self.l1c_spatial_bin_spacing_deg**2)),
            )

            # Check the repr exists
            assert "UltraPointingSet" in repr(ultra_pset)

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    def test_uneven_spacing_raises_error(self):
        """Test that uneven spacing in az/el raises ValueError"""

        # Create dataset with uneven az spacing
        uneven_az_dataset = xr.Dataset()
        uneven_az_dataset["epoch"] = 1
        uneven_az_dataset["azimuth_bin_center"] = np.array([0, 5, 15, 20, 30])
        uneven_az_dataset["elevation_bin_center"] = np.arange(5)

        with pytest.raises(ValueError, match="Azimuth bin spacing is not uniform"):
            ena_maps.UltraPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=uneven_az_dataset,
            )

        uneven_az_dataset["azimuth_bin_center"] = np.arange(5)
        uneven_az_dataset["elevation_bin_center"] = np.array([0, 5, 15, 20, 30])

        with pytest.raises(ValueError, match="Elevation bin spacing is not uniform"):
            ena_maps.UltraPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=uneven_az_dataset,
            )

        # Even but not the same spacing between az and el
        uneven_az_dataset["azimuth_bin_center"] = np.arange(5)
        uneven_az_dataset["elevation_bin_center"] = np.arange(5) * 2

        with pytest.raises(
            ValueError, match="Azimuth and elevation bin spacing do not match:"
        ):
            ena_maps.UltraPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=uneven_az_dataset,
            )


class TestRectangularSkyMap:
    @pytest.fixture(autouse=True)
    def _setup_ultra_l1c_pset_products(self, l1c_pset_products):
        """Setup fixture data as class attributes"""
        self.l1c_spatial_bin_spacing_deg = l1c_pset_products["spacing"]
        self.l1c_pset_products = l1c_pset_products["products"]
        self.ultra_psets = [
            ena_maps.UltraPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
            )
            for l1c_product in self.l1c_pset_products
        ]

    def test_instantiate(self):
        """Test instantiation of RectangularSkyMap"""
        rm = ena_maps.RectangularSkyMap(
            spacing_deg=2,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        # Check that the map is empty
        assert rm.data_dict == {}

        # Check that the reference frame is correctly set
        assert rm.spice_reference_frame == geometry.SpiceFrame.ECLIPJ2000

        # Check the number of points is (360/2) * (180/2)
        np.testing.assert_equal(rm.num_points, int(360 * 180 / 4))

        # Check the repr exists
        assert "RectangularSkyMap" in repr(rm)

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_project_pset_values_to_map_push_method(self, mock_frame_transform_az_el):
        """
        Test projection of PSET values to Rect. Map w "push" index matching method.

        If frame_transform_az_el is mocked to return the az and el unchanged, and the
        map has the same spacing as the PSETs, then the map should have
        the same values as the PSETs, summed.
        """
        index_matching_method = ena_maps.IndexMatchMethod.PUSH

        pset_spacing_deg = self.ultra_psets[0].spacing_deg

        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )

        rectangular_map = ena_maps.RectangularSkyMap(
            spacing_deg=pset_spacing_deg,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        # Project each PSET's values to the map (push method)
        for ultra_pset in self.ultra_psets:
            rectangular_map.project_pset_values_to_map(
                ultra_pset,
                value_keys=["counts", "exposure_time"],
                index_match_method=index_matching_method,
            )

        # Check that the map has been updated
        assert "counts" in rectangular_map.data_dict

        # Check that the map has the same values as the PSETs, summed
        simple_summed_pset_counts = np.zeros_like(rectangular_map.data_dict["counts"])
        for pset in self.l1c_pset_products:
            reshaped_pset_counts = pset["counts"].squeeze("epoch")
            # Reshape to the map's counts shape
            reshaped_pset_counts = reshaped_pset_counts.data.reshape(
                rectangular_map.data_dict["counts"].shape
            )
            simple_summed_pset_counts += reshaped_pset_counts

        np.testing.assert_array_equal(
            rectangular_map.data_dict["counts"],
            simple_summed_pset_counts,
        )

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    def test_project_pset_values_to_map_errors(self):
        index_matching_method = ena_maps.IndexMatchMethod.PUSH
        rectangular_map = ena_maps.RectangularSkyMap(
            spacing_deg=1,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        # An error should be raised if a key is not found in the PSET
        with pytest.raises(ValueError, match="Value key invalid not found"):
            rectangular_map.project_pset_values_to_map(
                self.ultra_psets[0],
                value_keys=["invalid"],
                index_match_method=index_matching_method,
            )

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_project_pset_values_to_map_pull_method(self, mock_frame_transform_az_el):
        """Test projection to Rect. Map fails w "pull" index matching method."""

        index_matching_method = ena_maps.IndexMatchMethod.PULL
        skymap_spacing = 10

        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )
        rectangular_map = ena_maps.RectangularSkyMap(
            spacing_deg=skymap_spacing,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        # Each map pixel will add the value of a single PSET pixel, so we'll start at 0
        # and add 0, 1, 2, 3, ... to the map
        expected_value_every_pixel = 0

        # Another way to test this is that (if the PSET pixels are
        # smaller than the SkyMap pixels) the sum of the counts in all PSETs should
        # be (PSET_spacing / SkyMap_spacing)^2 times the sum of the counts in the SkyMap
        total_pset_counts = np.zeros_like(self.ultra_psets[0].data["counts"].values)

        # Project each PSET's values to the map (pull method)
        for pset_num, ultra_pset in enumerate(self.ultra_psets):
            # Set the counts to be 0 in the first PSET, 1 in the second, etc.
            ultra_pset.data["counts"].values = np.full_like(
                ultra_pset.data["counts"].values, pset_num
            )

            rectangular_map.project_pset_values_to_map(
                ultra_pset,
                value_keys=["counts", "exposure_time"],
                index_match_method=index_matching_method,
            )
            expected_value_every_pixel += pset_num

            total_pset_counts += ultra_pset.data["counts"].values

        # Check that the map has been updated
        assert "counts" in rectangular_map.data_dict

        np.testing.assert_allclose(
            rectangular_map.data_dict["counts"],
            expected_value_every_pixel,
        )
        downsample_ratio = skymap_spacing / self.l1c_spatial_bin_spacing_deg
        np.testing.assert_allclose(
            rectangular_map.data_dict["counts"].sum(),
            total_pset_counts.sum() / (downsample_ratio**2),
        )


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

        # Check that the map's az/el points at the matched indices
        # are the same as the input az/el points to within the spacing of the map
        matched_map_az_el = mock_rect_map.az_el_points[
            flat_indices_input_grid_output_frame
        ]
        np.testing.assert_allclose(
            matched_map_az_el[:, 0],
            mock_pset_input_frame.az_el_points[:, 0],
            atol=np.deg2rad(map_spacing_deg),
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

    def test_match_coords_to_indices_pset_to_pset_error(self):
        mock_pset_input_frame = ena_maps.UltraPointingSet(
            l1c_dataset=self.l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        mock_pset_output_frame = ena_maps.UltraPointingSet(
            l1c_dataset=self.l1c_pset_products[1],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        with pytest.raises(
            ValueError, match="Cannot match indices between two PointingSet objects"
        ):
            ena_maps.match_coords_to_indices(
                mock_pset_input_frame, mock_pset_output_frame
            )

    def test_match_coords_to_indices_map_to_map_no_et_error(self):
        mock_rect_map = ena_maps.RectangularSkyMap(
            spacing_deg=2,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        mock_other_map = ena_maps.RectangularSkyMap(
            spacing_deg=4,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        with pytest.raises(
            ValueError,
            match="Event time must be specified if both objects are SkyMaps.",
        ):
            ena_maps.match_coords_to_indices(mock_rect_map, mock_other_map)

        # No error if event time is specified
        _ = ena_maps.match_coords_to_indices(
            mock_rect_map, mock_other_map, event_time=0
        )
