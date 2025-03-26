"""Test classes and methods in ena_maps.py."""

from __future__ import annotations

from copy import deepcopy
from unittest import mock

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import xarray as xr

from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils import spatial_utils
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry


@pytest.fixture(autouse=True, scope="module")
def setup_all_pset_products(ultra_l1c_pset_datasets, rectangular_l1c_pset_datasets):
    """
    Setup fixture data once for all tests.

    This is relatively computationally intensive for the high resolution PSETs,
    so we use a module-level fixture to avoid repeating the setup code. However,
    some tests need to modify the PSETs, so we use a function-level fixture to
    make a deepcopy of the PSETs for each test function.
    """
    hp_ultra_nside = ultra_l1c_pset_datasets["nside"]
    hp_ultra_l1c_pset_products = ultra_l1c_pset_datasets["products"]
    rect_spacing = rectangular_l1c_pset_datasets["spacing"]
    rect_rectangular_l1c_pset_products = rectangular_l1c_pset_datasets["products"]
    return {
        "hp_ultra_nside": hp_ultra_nside,
        "hp_ultra_l1c_pset_products": hp_ultra_l1c_pset_products,
        "rect_spacing": rect_spacing,
        "rect_rectangular_l1c_pset_products": rect_rectangular_l1c_pset_products,
    }


class TestUltraPointingSet:
    @pytest.fixture(autouse=True)
    def _setup_ultra_l1c_pset_products(self, setup_all_pset_products):
        """Setup fixture data as class attributes"""
        self.nside = setup_all_pset_products["hp_ultra_nside"]
        self.l1c_pset_products = deepcopy(
            setup_all_pset_products["hp_ultra_l1c_pset_products"]
        )

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
            # Check tiling is HEALPix
            assert ultra_pset.tiling_type is ena_maps.SkyTilingType.HEALPIX

            # Check that the reference frame is correctly set
            assert ultra_pset.spice_reference_frame is geometry.SpiceFrame.IMAP_DPS

            # Check the number of points is (360/0.5) * (180/0.5)
            np.testing.assert_equal(
                ultra_pset.num_points,
                hp.nside2npix(self.nside),
            )

            # Check the repr exists
            assert "UltraPointingSet" in repr(ultra_pset)

            # Checks for the property methods:
            # Check that the unwrapped_dims_dict is as expected
            assert ultra_pset.unwrapped_dims_dict["counts"] == (
                "epoch",
                "energy",
                "pixel",
            )
            # Check the non_spatial_coords are as expected
            assert tuple(ultra_pset.non_spatial_coords.keys()) == (
                "epoch",
                "energy",
            )

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    def test_different_spacing_raises_error(self):
        """Test that different spaced az/el from the L1C dataset raises ValueError"""

        ultra_pset_ds = self.l1c_pset_products[0]
        # Modify the dataset to have different spacing
        ultra_pset_ds[CoordNames.ELEVATION_L1C.value].values = np.arange(
            ultra_pset_ds[CoordNames.ELEVATION_L1C.value].size
        )

        with pytest.raises(ValueError, match="do not match"):
            ena_maps.UltraPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=ultra_pset_ds,
            )


class TestRectangularSkyMap:
    @pytest.fixture(autouse=True)
    def _setup_ultra_l1c_pset_products(self, setup_all_pset_products):
        """Setup fixture data as class attributes"""
        self.ultra_l1c_nside = setup_all_pset_products["hp_ultra_nside"]
        self.ultra_l1c_pset_products = deepcopy(
            setup_all_pset_products["hp_ultra_l1c_pset_products"]
        )
        self.ultra_psets = [
            ena_maps.UltraPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
            )
            for l1c_product in self.ultra_l1c_pset_products
        ]

    @pytest.fixture(autouse=True)
    def _setup_rectangular_l1c_pset_products(self, setup_all_pset_products):
        """Setup fixture data as class attributes"""
        self.rectangular_l1c_spacing_deg = setup_all_pset_products["rect_spacing"]
        self.rectangular_l1c_pset_products = deepcopy(
            setup_all_pset_products["rect_rectangular_l1c_pset_products"]
        )
        self.rectangular_psets = [
            ena_maps.RectangularPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
            )
            for l1c_product in self.rectangular_l1c_pset_products
        ]

    def test_instantiate(self):
        """Test instantiation of RectangularSkyMap"""
        rm = ena_maps.RectangularSkyMap(
            spacing_deg=2,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        # Check that the map data is an empty xarray Dataset
        assert isinstance(rm.data_1d, xr.Dataset)
        assert rm.data_1d.data_vars == {}

        # Check that the reference frame is correctly set
        assert rm.spice_reference_frame == geometry.SpiceFrame.ECLIPJ2000

        # Check the number of points is (360/2) * (180/2)
        np.testing.assert_equal(rm.num_points, int(360 * 180 / 4))

        # Check the repr exists
        assert "RectangularSkyMap" in repr(rm)

        np.testing.assert_array_equal(
            rm.binning_grid_shape, (360 / rm.spacing_deg, 180 / rm.spacing_deg)
        )

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_project_healpix_pset_values_to_map_push_method(
        self, mock_frame_transform_az_el
    ):
        """
        Test projection of Healpix tiled PSET values to RectMap w "push" index matching.

        If frame_transform_az_el is mocked to return the az and el unchanged,
        then the map should have the same total counts in each energy bin
        as the PSETs, summed.
        """
        index_matching_method = ena_maps.IndexMatchMethod.PUSH

        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )

        rectangular_map = ena_maps.RectangularSkyMap(
            spacing_deg=2,
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
        assert "counts" in rectangular_map.data_1d.data_vars

        # Check that the map has the same values as the PSETs, summed
        simple_summed_pset_counts_by_energy = np.zeros(
            shape=(
                self.ultra_l1c_pset_products[0]["counts"].sizes[
                    CoordNames.ENERGY.value
                ],
            )
        )
        for pset in self.ultra_l1c_pset_products:
            simple_summed_pset_counts_by_energy += pset["counts"].sum(
                dim=[d for d in pset["counts"].dims if d != CoordNames.ENERGY.value]
            )

        rmap_counts_per_energy_bin = rectangular_map.data_1d["counts"].sum(
            dim=[
                d
                for d in rectangular_map.data_1d["counts"].dims
                if d != CoordNames.ENERGY.value
            ]
        )

        np.testing.assert_array_equal(
            rmap_counts_per_energy_bin,
            simple_summed_pset_counts_by_energy,
        )

    @pytest.mark.usefixtures("_setup_rectangular_l1c_pset_products")
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_project_rect_pset_values_to_map_push_method(
        self, mock_frame_transform_az_el
    ):
        """
        Test projection of Rect PSET values to Rect Map w "push" index matching method.

        If frame_transform_az_el is mocked to return the az and el unchanged, and the
        map has the same spacing as the PSETs, then the map should have
        the same values as the PSETs, summed.
        """
        index_matching_method = ena_maps.IndexMatchMethod.PUSH

        pset_spacing_deg = self.rectangular_psets[0].spacing_deg

        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )

        rectangular_map = ena_maps.RectangularSkyMap(
            spacing_deg=pset_spacing_deg,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        # Project each PSET's values to the map (push method)
        for rectangular_pset in self.rectangular_psets:
            rectangular_map.project_pset_values_to_map(
                rectangular_pset,
                value_keys=["counts", "exposure_time"],
                index_match_method=index_matching_method,
            )

        # Check that the map has been updated
        assert "counts" in rectangular_map.data_1d.data_vars

        # Check that the map has the same values as the PSETs, summed
        simple_summed_pset_counts_by_energy = np.zeros(
            shape=(
                self.rectangular_l1c_pset_products[0]["counts"].sizes[
                    CoordNames.ENERGY.value
                ],
            )
        )
        for pset in self.rectangular_l1c_pset_products:
            simple_summed_pset_counts_by_energy += pset["counts"].sum(
                dim=[d for d in pset["counts"].dims if d != CoordNames.ENERGY.value]
            )

        rmap_counts_per_energy_bin = rectangular_map.data_1d["counts"].sum(
            dim=[
                d
                for d in rectangular_map.data_1d["counts"].dims
                if d != CoordNames.ENERGY.value
            ]
        )

        np.testing.assert_array_equal(
            rmap_counts_per_energy_bin,
            simple_summed_pset_counts_by_energy,
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

    @pytest.mark.usefixtures("_setup_rectangular_l1c_pset_products")
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_project_rect_pset_values_to_map_pull_method(
        self, mock_frame_transform_az_el
    ):
        """
        Test projection Rect PSET to Rect. Map with "pull" index matching method.

        NOTE: Pull index matching is only expected to be done with Rectangularly tiled
        PointingSet objects.
        """

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
        total_pset_counts = np.zeros_like(
            self.rectangular_l1c_pset_products[0]["counts"].values
        )

        # Project each PSET's values to the map (pull method)
        for pset_num, rectangular_pset in enumerate(self.rectangular_psets):
            # Set the counts to be 0 in the first PSET, 1 in the second, etc.
            rectangular_pset.data["counts"].values = np.full_like(
                rectangular_pset.data["counts"].values, pset_num
            )

            rectangular_map.project_pset_values_to_map(
                rectangular_pset,
                value_keys=["counts", "exposure_time"],
                index_match_method=index_matching_method,
            )
            expected_value_every_pixel += pset_num

            total_pset_counts += rectangular_pset.data["counts"].values

        # Check that the map has been updated
        assert "counts" in rectangular_map.data_1d

        np.testing.assert_allclose(
            rectangular_map.data_1d["counts"],
            expected_value_every_pixel,
        )
        downsample_ratio = skymap_spacing / self.rectangular_l1c_spacing_deg
        np.testing.assert_allclose(
            rectangular_map.data_1d["counts"].sum(),
            total_pset_counts.sum() / (downsample_ratio**2),
        )

        # Convert to xarray Dataset and check the data is as expected
        # This is a method, which could be tested separately, but that would be
        # innefficient, as it would require all the same, computationally intensive
        # operations to be repeated as this test
        rect_map_ds = rectangular_map.to_dataset()
        assert "counts" in rect_map_ds.data_vars
        assert rect_map_ds["counts"].shape == (
            1,
            rectangular_pset.data["counts"].sizes[CoordNames.ENERGY.value],
            360 / skymap_spacing,
            180 / skymap_spacing,
        )
        assert rect_map_ds["counts"].dims == (
            CoordNames.TIME.value,
            CoordNames.ENERGY.value,
            CoordNames.AZIMUTH_L2.value,
            CoordNames.ELEVATION_L2.value,
        )

        # Check that the data is as expected
        np.testing.assert_array_equal(
            rect_map_ds["counts"].values,
            spatial_utils.rewrap_even_spaced_az_el_grid(
                rectangular_map.data_1d["counts"].values,
                rectangular_map.binning_grid_shape,
            ),
        )


class TestHealpixSkyMap:
    @pytest.fixture(autouse=True)
    def _setup_ultra_l1c_pset_products(self, setup_all_pset_products):
        """Setup fixture data as class attributes"""
        self.ultra_l1c_nside = setup_all_pset_products["hp_ultra_nside"]
        self.ultra_l1c_pset_products = deepcopy(
            setup_all_pset_products["hp_ultra_l1c_pset_products"]
        )
        self.ultra_psets = [
            ena_maps.UltraPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
            )
            for l1c_product in self.ultra_l1c_pset_products
        ]

    @pytest.fixture(autouse=True)
    def _setup_rectangular_l1c_pset_products(self, setup_all_pset_products):
        """Setup fixture data as class attributes"""
        self.rectangular_l1c_spacing_deg = setup_all_pset_products["rect_spacing"]
        self.rectangular_l1c_pset_products = deepcopy(
            setup_all_pset_products["rect_rectangular_l1c_pset_products"]
        )
        self.rectangular_psets = [
            ena_maps.RectangularPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
            )
            for l1c_product in self.rectangular_l1c_pset_products
        ]

    @pytest.mark.parametrize(
        "nside",
        [8, 16, 32],
    )
    @pytest.mark.parametrize("nested", [True, False], ids=["nested", "ring"])
    def test_instantiate(self, nside, nested):
        """Test instantiation of HealpixSkyMap"""
        hp_map = ena_maps.HealpixSkyMap(
            nside=nside,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
            nested=nested,
        )

        # Check that the map data is an empty xarray Dataset
        assert isinstance(hp_map.data_1d, xr.Dataset)
        assert hp_map.data_1d.data_vars == {}

        # Check that the reference frame is correctly set
        assert hp_map.spice_reference_frame is geometry.SpiceFrame.ECLIPJ2000
        # Check that the nside and nested properties are set correctly
        np.testing.assert_equal(hp_map.nside, nside)
        np.testing.assert_equal(hp_map.nested, nested)
        # Check the number of points is 12 * nside^2
        np.testing.assert_equal(hp_map.num_points, 12 * nside**2)
        # There will be az, el values for each pixel
        assert hp_map.az_el_points.shape == (hp_map.num_points, 2)
        # The az must be in the range [0, 360) degrees
        # and el in the range [-90, 90)
        assert np.all(hp_map.az_el_points[:, 0] >= 0)
        assert np.all(hp_map.az_el_points[:, 0] < 360)
        assert np.all(hp_map.az_el_points[:, 1] >= -90)
        assert np.all(hp_map.az_el_points[:, 1] < 90)

        # Check that the binning grid shape is just a tuple of num_points
        np.testing.assert_equal(hp_map.binning_grid_shape, (hp_map.num_points,))

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    @pytest.mark.parametrize(
        "nside,degree_tolerance",
        [
            (8, 6),
            (16, 3),
            (32, 2),
        ],
    )
    @pytest.mark.parametrize("nested", [True, False], ids=["nested", "ring"])
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_project_healpix_pset_values_to_map_push_method(
        self, mock_frame_transform_az_el, nside, degree_tolerance, nested
    ):
        """
        Test that PointingSet which contains bright spot pushes to correct spot in map.

        Parameterized over nside (of the map, not the PSET), nested.
        The tolerance for lower nsides must be higher because the
        Healpix pixels are larger.
        """

        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )

        index_matching_method = ena_maps.IndexMatchMethod.PUSH

        # Create a PointingSet with a bright spot
        mock_pset_input_frame = ena_maps.UltraPointingSet(
            l1c_dataset=self.ultra_l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        mock_pset_input_frame.data["counts"].values = np.zeros_like(
            mock_pset_input_frame.data["counts"].values
        )

        input_bright_pixel_number = hp.ang2pix(
            nside=mock_pset_input_frame.nside,
            theta=180,
            phi=0,
            nest=mock_pset_input_frame.nested,
            lonlat=True,
        )
        input_bright_pixel_az_el_deg = mock_pset_input_frame.az_el_points[
            input_bright_pixel_number
        ]
        mock_pset_input_frame.data["counts"].values[
            :,
            :,
            input_bright_pixel_number,
        ] = 1

        # Create a Healpix map
        hp_map = ena_maps.HealpixSkyMap(
            nside=nside,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
            nested=nested,
        )

        # Project the PointingSet to the Healpix map
        hp_map.project_pset_values_to_map(
            mock_pset_input_frame,
            value_keys=[
                "counts",
            ],
            index_match_method=index_matching_method,
        )

        # Check that the map has been updated
        assert "counts" in hp_map.data_1d.data_vars

        # Find the maximum value in the spatial pixel dimension of the healpix map
        bright_hp_pixel_index = hp_map.data_1d["counts"][0, :].argmax()
        bright_hp_pixel_az_el = hp_map.az_el_points[bright_hp_pixel_index]

        np.testing.assert_allclose(
            bright_hp_pixel_az_el,
            input_bright_pixel_az_el_deg,
            atol=degree_tolerance,
        )

    @pytest.mark.usefixtures("_setup_rectangular_l1c_pset_products")
    @pytest.mark.parametrize(
        "nside,degree_tolerance",
        [
            (8, 6),
            (16, 3),
            (32, 2),
        ],
    )
    @pytest.mark.parametrize("nested", [True, False], ids=["nested", "ring"])
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_project_rect_pset_values_to_map_push_method(
        self, mock_frame_transform_az_el, nside, degree_tolerance, nested
    ):
        """
        Test that PointingSet which contains bright spot pushes to correct spot in map.

        Parameterized over nside, nested. The tolerance for lower nsides must be higher
        because the Healpix pixels are larger.
        """

        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )

        index_matching_method = ena_maps.IndexMatchMethod.PUSH

        # Create a PointingSet with a bright spot
        mock_pset_input_frame = ena_maps.RectangularPointingSet(
            l1c_dataset=self.rectangular_l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        mock_pset_input_frame.data["counts"].values = np.zeros_like(
            mock_pset_input_frame.data["counts"].values
        )

        input_bright_pixel_az_el_deg = (110, 55)
        mock_pset_input_frame.data["counts"].values[
            :,
            :,
            int(input_bright_pixel_az_el_deg[0] // mock_pset_input_frame.spacing_deg),
            int(
                (90 + input_bright_pixel_az_el_deg[1])
                // mock_pset_input_frame.spacing_deg
            ),
        ] = 1

        # Create a Healpix map
        hp_map = ena_maps.HealpixSkyMap(
            nside=nside,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
            nested=nested,
        )

        # Project the PointingSet to the Healpix map
        hp_map.project_pset_values_to_map(
            mock_pset_input_frame,
            value_keys=[
                "counts",
            ],
            index_match_method=index_matching_method,
        )

        # Check that the map has been updated
        assert "counts" in hp_map.data_1d.data_vars

        # Find the maximum value in the spatial pixel dimension of the healpix map
        bright_hp_pixel_index = hp_map.data_1d["counts"][0, 0].argmax(dim="pixel")
        bright_hp_pixel_az_el = hp_map.az_el_points[bright_hp_pixel_index]

        np.testing.assert_allclose(
            bright_hp_pixel_az_el,
            input_bright_pixel_az_el_deg,
            atol=degree_tolerance,
        )

        # Convert to xarray Dataset and check the data is as expected
        hp_map_ds = hp_map.to_dataset()
        assert "counts" in hp_map_ds.data_vars
        assert hp_map_ds["counts"].shape == (
            1,
            mock_pset_input_frame.data["counts"].sizes[CoordNames.ENERGY.value],
            hp_map.num_points,
        )
        assert hp_map_ds["counts"].dims == (
            CoordNames.TIME.value,
            CoordNames.ENERGY.value,
            CoordNames.HEALPIX_INDEX.value,
        )
        np.testing.assert_array_equal(
            hp_map_ds["counts"].values,
            hp_map.data_1d["counts"].values,
        )


class TestIndexMatching:
    @pytest.fixture(autouse=True)
    def _setup_rectangular_l1c_pset_products(self, setup_all_pset_products):
        """Setup fixture data as class attributes"""
        self.rectangular_l1c_spacing_deg = setup_all_pset_products["rect_spacing"]
        self.rectangular_l1c_pset_products = deepcopy(
            setup_all_pset_products["rect_rectangular_l1c_pset_products"]
        )
        self.rectangular_psets = [
            ena_maps.RectangularPointingSet(
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
                l1c_dataset=l1c_product,
            )
            for l1c_product in self.rectangular_l1c_pset_products
        ]

    @pytest.mark.parametrize(
        "map_spacing_deg",
        [0.5, 1, 10],
    )
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_match_coords_to_indices_rect_pset_to_rect_map(
        self, mock_frame_transform_az_el, map_spacing_deg
    ):
        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )

        # Mock a PSET, overriding the az/el points
        mock_pset_input_frame = ena_maps.RectangularPointingSet(
            l1c_dataset=self.rectangular_l1c_pset_products[0],
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
        mock_pset_input_frame.az_el_points = manual_az_el_coords

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

        # Create the rectangular map and check the output values
        rect_map = ena_maps.RectangularSkyMap(
            spacing_deg=map_spacing_deg,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        flat_indices_input_grid_output_frame = ena_maps.match_coords_to_indices(
            mock_pset_input_frame, rect_map
        )
        assert rect_map.num_points == 360 * 180 / map_spacing_deg**2
        assert len(flat_indices_input_grid_output_frame) == len(manual_az_el_coords)
        np.testing.assert_equal(
            flat_indices_input_grid_output_frame, expected_output_pixel
        )

        # Check that the map's az/el points at the matched indices
        # are the same as the input az/el points to within the spacing of the map
        matched_map_az_el = rect_map.az_el_points[flat_indices_input_grid_output_frame]
        np.testing.assert_allclose(
            matched_map_az_el[:, 0],
            mock_pset_input_frame.az_el_points[:, 0],
            atol=map_spacing_deg,
        )

    @pytest.mark.parametrize(
        "nside,degree_tolerance",
        [
            (8, 12),
            (16, 6),
            (32, 3),
        ],
        ids=["nside8", "nside16", "nside32"],
    )
    @pytest.mark.parametrize("nested", [True, False], ids=["nested", "ring"])
    @mock.patch("imap_processing.spice.geometry.frame_transform_az_el")
    def test_match_coords_to_indices_rect_pset_to_healpix_map(
        self, mock_frame_transform_az_el, nside, degree_tolerance, nested
    ):
        # Mock frame_transform to return the az and el unchanged
        mock_frame_transform_az_el.side_effect = (
            lambda et, az_el, from_frame, to_frame, degrees: az_el
        )
        hp_map = ena_maps.HealpixSkyMap(
            nside=nside, spice_frame=geometry.SpiceFrame.ECLIPJ2000, nested=nested
        )

        # Make a PointingSet
        mock_pset_input_frame = ena_maps.RectangularPointingSet(
            l1c_dataset=self.rectangular_l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )

        # Match the PSET to the Healpix map
        healpix_indices_of_rect_pixels = ena_maps.match_coords_to_indices(
            mock_pset_input_frame, hp_map
        )

        # Check that the map's az/el points at the matched indices
        # are the same as the input az/el points to within degree_tolerance,
        # but we must ignore the polar regions and azimuthal wrap-around regions
        rect_equatorial_elevations_mask = (
            np.abs(mock_pset_input_frame.az_el_points[:, 1]) < 60
        )
        rect_az_non_wraparound_mask = (
            mock_pset_input_frame.az_el_points[:, 0] < 340
        ) & (mock_pset_input_frame.az_el_points[:, 0] > 20)
        rect_good_az_el_mask = (
            rect_equatorial_elevations_mask & rect_az_non_wraparound_mask
        )
        matched_map_az_el = np.column_stack(
            hp.pix2ang(
                nside=nside,
                ipix=healpix_indices_of_rect_pixels,
                nest=nested,
                lonlat=True,
            )
        )
        np.testing.assert_allclose(
            matched_map_az_el[rect_good_az_el_mask, 0],
            mock_pset_input_frame.az_el_points[rect_good_az_el_mask, 0],
            atol=degree_tolerance,
        )

    def test_match_coords_to_indices_pset_to_invalid_map(
        self,
    ):
        mock_pset_input_frame = ena_maps.RectangularPointingSet(
            l1c_dataset=self.rectangular_l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        # Until implemented, just change the tiling on a RectangularSkyMap
        mock_invalid_map = ena_maps.RectangularSkyMap(
            spacing_deg=2,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        mock_invalid_map.tiling_type = "INVALID"

        # Should raise ValueError if the tiling type is invalid
        with pytest.raises(ValueError, match="Tiling type of the output frame"):
            ena_maps.match_coords_to_indices(mock_pset_input_frame, mock_invalid_map)

    def test_match_coords_to_indices_pset_to_pset_error(self):
        mock_pset_input_frame = ena_maps.RectangularPointingSet(
            l1c_dataset=self.rectangular_l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        mock_pset_output_frame = ena_maps.RectangularPointingSet(
            l1c_dataset=self.rectangular_l1c_pset_products[1],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        with pytest.raises(
            ValueError, match="Cannot match indices between two PointingSet objects"
        ):
            ena_maps.match_coords_to_indices(
                mock_pset_input_frame, mock_pset_output_frame
            )

    def test_match_coords_to_indices_map_to_map_no_et_error(self):
        mock_rect_map_1 = ena_maps.RectangularSkyMap(
            spacing_deg=2,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        mock_rect_map_2 = ena_maps.RectangularSkyMap(
            spacing_deg=4,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )
        with pytest.raises(
            ValueError,
            match="Event time must be specified if both objects are SkyMaps.",
        ):
            ena_maps.match_coords_to_indices(mock_rect_map_1, mock_rect_map_2)

        # No error if event time is specified
        _ = ena_maps.match_coords_to_indices(
            mock_rect_map_1, mock_rect_map_2, event_et=0
        )
