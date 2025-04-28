"""Test classes and methods in ena_maps.py."""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest import mock

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf, write_cdf
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
                l1c_product,
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
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
                "energy_bin_geometric_mean",
                "pixel",
            )
            # Check the non_spatial_coords are as expected
            assert tuple(ultra_pset.non_spatial_coords.keys()) == (
                "epoch",
                "energy_bin_geometric_mean",
            )

    @pytest.mark.usefixtures("_setup_ultra_l1c_pset_products")
    def test_init_cdf(
        self,
    ):
        ultra_pset = self.l1c_pset_products[0]

        cdf_filepath = write_cdf(ultra_pset, istp=False)

        ultra_pset_from_dataset = ena_maps.UltraPointingSet(ultra_pset)

        ultra_pset_from_str = ena_maps.UltraPointingSet(cdf_filepath)
        ultra_pset_from_path = ena_maps.UltraPointingSet(Path(cdf_filepath))

        np.testing.assert_allclose(
            ultra_pset_from_dataset.data["counts"].values,
            ultra_pset_from_str.data["counts"].values,
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            ultra_pset_from_dataset.data["counts"].values,
            ultra_pset_from_path.data["counts"].values,
            rtol=1e-6,
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
                ultra_pset_ds,
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
            )


@pytest.fixture(scope="module")
def hi_pset_cdf_path(imap_tests_path):
    return imap_tests_path / "hi/data/l1/imap_hi_l1c_45sensor-pset_20250415_v999.cdf"


@pytest.mark.external_test_data
class TestHiPointingSet:
    """Test suite for HiPointingSet class."""

    def test_init(self, hi_pset_cdf_path):
        """Test coverage for __init__ method."""
        pset_ds = load_cdf(hi_pset_cdf_path)
        hi_pset = ena_maps.HiPointingSet(pset_ds)
        assert isinstance(hi_pset, ena_maps.HiPointingSet)
        assert hi_pset.spice_reference_frame == geometry.SpiceFrame.ECLIPJ2000
        assert hi_pset.num_points == 3600
        np.testing.assert_array_equal(hi_pset.az_el_points.shape, (3600, 2))

    def test_from_cdf(self, hi_pset_cdf_path):
        """Test coverage for from_cdf method."""
        hi_pset = ena_maps.HiPointingSet(hi_pset_cdf_path)
        assert isinstance(hi_pset, ena_maps.HiPointingSet)

    def test_plays_nice_with_rectangular_sky_map(self, hi_pset_cdf_path):
        """Test that HiPointingSet works with RectangularSkyMap"""
        hi_pset = ena_maps.HiPointingSet(hi_pset_cdf_path)
        rect_map = ena_maps.RectangularSkyMap(
            spacing_deg=2, spice_frame=geometry.SpiceFrame.ECLIPJ2000
        )
        rect_map.project_pset_values_to_map(hi_pset, ["counts", "exposure_times"])
        assert rect_map.data_1d["counts"].max() > 0


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
                l1c_product,
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
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
                l1c_product,
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
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
                value_keys=["counts", "exposure_factor"],
                index_match_method=index_matching_method,
            )

        # Check that the map has been updated
        assert "counts" in rectangular_map.data_1d.data_vars

        # Check that the map has the same values as the PSETs, summed
        simple_summed_pset_counts_by_energy = np.zeros(
            shape=(
                self.ultra_l1c_pset_products[0]["counts"].sizes[
                    CoordNames.ENERGY_ULTRA.value
                ],
            )
        )
        for pset in self.ultra_l1c_pset_products:
            simple_summed_pset_counts_by_energy += pset["counts"].sum(
                dim=[
                    d for d in pset["counts"].dims if d != CoordNames.ENERGY_ULTRA.value
                ]
            )

        rmap_counts_per_energy_bin = rectangular_map.data_1d["counts"].sum(
            dim=[
                d
                for d in rectangular_map.data_1d["counts"].dims
                if d != CoordNames.ENERGY_ULTRA.value
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

        pset_spacing_deg = self.rectangular_psets[0].sky_grid.spacing_deg

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
                value_keys=["counts", "exposure_factor"],
                index_match_method=index_matching_method,
            )

        # Check that the map has been updated
        assert "counts" in rectangular_map.data_1d.data_vars

        # Check that the map has the same values as the PSETs, summed
        simple_summed_pset_counts_by_energy = np.zeros(
            shape=(
                self.rectangular_l1c_pset_products[0]["counts"].sizes[
                    CoordNames.ENERGY_ULTRA.value
                ],
            )
        )
        for pset in self.rectangular_l1c_pset_products:
            simple_summed_pset_counts_by_energy += pset["counts"].sum(
                dim=[
                    d for d in pset["counts"].dims if d != CoordNames.ENERGY_ULTRA.value
                ]
            )

        rmap_counts_per_energy_bin = rectangular_map.data_1d["counts"].sum(
            dim=[
                d
                for d in rectangular_map.data_1d["counts"].dims
                if d != CoordNames.ENERGY_ULTRA.value
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
                value_keys=["counts", "exposure_factor"],
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
            rectangular_pset.data["counts"].sizes[CoordNames.ENERGY_ULTRA.value],
            360 / skymap_spacing,
            180 / skymap_spacing,
        )
        assert rect_map_ds["counts"].dims == (
            CoordNames.TIME.value,
            CoordNames.ENERGY_ULTRA.value,
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
                l1c_product,
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
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
                l1c_product,
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
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
            self.ultra_l1c_pset_products[0],
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
        bright_hp_pixel_index = hp_map.data_1d["counts"][0, :].values.argmax()
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
            self.rectangular_l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        mock_pset_input_frame.data["counts"].values = np.zeros_like(
            mock_pset_input_frame.data["counts"].values
        )

        input_bright_pixel_az_el_deg = (110, 55)
        mock_pset_input_frame.data["counts"].values[
            :,
            :,
            int(
                input_bright_pixel_az_el_deg[0]
                // mock_pset_input_frame.sky_grid.spacing_deg
            ),
            int(
                (90 + input_bright_pixel_az_el_deg[1])
                // mock_pset_input_frame.sky_grid.spacing_deg
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
            mock_pset_input_frame.data["counts"].sizes[CoordNames.ENERGY_ULTRA.value],
            hp_map.num_points,
        )
        assert hp_map_ds["counts"].dims == (
            CoordNames.TIME.value,
            CoordNames.ENERGY_ULTRA.value,
            CoordNames.HEALPIX_INDEX.value,
        )
        np.testing.assert_array_equal(
            hp_map_ds["counts"].values,
            hp_map.data_1d["counts"].values,
        )

    @mock.patch("astropy_healpix.healpy.ang2pix")
    def test_calculate_rect_pixel_value_from_healpix_map_n_subdivisions(
        self,
        mock_ang2pix,
    ):
        """Test getting rectangular pixel values from HealpixSkyMap via subdivision."""

        # Mock ang2pix to return fixed values based on a dict
        pixel_dict = {
            # 0 subdiv - just 1 pixel
            (180, 0): 0,
            # 1 subdiv - all subpix have same solid angle because centered on equator
            (179, -1): 1,
            (179, 1): 2,
            (181, -1): 3,
            (181, 1): 4,
            # 2 subdiv - 'Inner' subpix have larger solid angle than 'outer' subpix
            (178.5, -1.5): 5,
            (178.5, -0.5): 6,
            (178.5, 0.5): 7,
            (178.5, 1.5): 8,
            (179.5, -1.5): 9,
            (179.5, -0.5): 10,
            (179.5, 0.5): 11,
            (179.5, 1.5): 12,
            (180.5, -1.5): 12,
            (180.5, -0.5): 14,
            (180.5, 0.5): 15,
            (180.5, 1.5): 16,
            (181.5, -1.5): 17,
            (181.5, -0.5): 18,
            (181.5, 0.5): 19,
            (181.5, 1.5): 20,
        }
        expected_mean_0_subdivisions = 0
        expected_mean_1_subdivisions = 2.5
        expected_mean_2_subdivisions = 12.5

        def mock_ang2pix_fn(nside, theta, phi, nest=True, lonlat=False):
            vals = []
            for pix_num in range(len(theta)):
                key = (theta[pix_num], phi[pix_num])
                vals.append(pixel_dict.get(key, 0))
            return np.array(vals)

        hp_map = ena_maps.HealpixSkyMap(
            nside=16,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
            nested=True,
        )
        hp_map.data_1d["counts"] = xr.DataArray(
            data=[
                np.arange(hp_map.num_points),
            ],
            dims=["epoch", "pixel"],
        )

        for num_subdiv, (expected_value, atol) in enumerate(
            [
                # The first subdivs have all the same solid angle
                (expected_mean_0_subdivisions, 1e-9),
                (expected_mean_1_subdivisions, 1e-9),
                # Slight difference from not taking into account asym solid angle
                (expected_mean_2_subdivisions, 0.1),
            ]
        ):
            mock_ang2pix.reset_mock()
            mock_ang2pix.side_effect = mock_ang2pix_fn
            mean_value = (
                hp_map.calculate_rect_pixel_value_from_healpix_map_n_subdivisions(
                    rect_pix_center_lon_lat=(180, 0),
                    rect_pix_spacing_deg=4,
                    value_array=hp_map.data_1d["counts"],
                    num_subdivisions=num_subdiv,
                )
            )
            np.testing.assert_allclose(
                mean_value,
                expected_value,
                atol=atol,
                err_msg=f"Failed for num_subdivisions: {num_subdiv}",
            )
        hp_map.calculate_rect_pixel_value_from_healpix_map_n_subdivisions(
            rect_pix_center_lon_lat=(180, 0),
            rect_pix_spacing_deg=2,
            value_array=hp_map.data_1d["counts"],
            num_subdivisions=0,
        )

    @mock.patch(
        "imap_processing.ena_maps.ena_maps.HealpixSkyMap.calculate_rect_pixel_value_from_healpix_map_n_subdivisions"
    )
    def test_get_rect_pixel_value_recursive_subdivs(
        self,
        mock_calculate_rect_pixel_value_from_healpix_map_n_subdivisions,
    ):
        """Test that the recursive subdivision works as expected with different rtol."""

        # Mock the function to return a fixed value for a number of subdivisions
        value_by_subdivisions = {
            0: 100.0,
            1: 110.0,  # 10/110 = 0.09090909 change
            2: 105.0,  # 5/105 = 0.04761905 change
            3: 107.0,  # 2/107 = 0.01869159 change
            4: 107.5,  # 0.5/107.5 = 0.00465116 change
            5: 107.51,  # 0.01/107.51 = 0.00009301 change
            6: 107.5099,  # 0.0001/107.5099 = 0.00000093 change
            7: 120,  # Big change - but will stop because of MAX SUBDIVS
        }
        required_rtols = [
            0.1,
            0.05,
            0.02,
            0.005,
            0.0001,
            0.000001,
            1e-12,
        ]

        mock_calculate_rect_pixel_value_from_healpix_map_n_subdivisions.side_effect = (
            lambda *args, **kwargs: np.array(
                [
                    value_by_subdivisions[kwargs["num_subdivisions"]],
                ]
            )
        )
        hp_map = ena_maps.HealpixSkyMap(
            nside=16,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        # Test the recursive subdivision by setting different tolerances to get the
        # expected number of subdivisions and resultant mean value.
        for expected_subdiv_level in range(1, len(required_rtols)):
            mean, depth = hp_map.get_rect_pixel_value_recursive_subdivs(
                rect_pix_center_lon_lat=(180, 0),
                rect_pix_spacing_deg=4,
                value_array=[],
                rtol=required_rtols[expected_subdiv_level - 1],
                max_subdivision_depth=7,
            )
            assert depth == expected_subdiv_level
            np.testing.assert_equal(
                mean,
                value_by_subdivisions[expected_subdiv_level],
                err_msg=f"Failed for expected_subdiv_level: {expected_subdiv_level}",
            )

    def test_to_rectangular_skymap(
        self,
    ):
        hp_map = ena_maps.HealpixSkyMap(
            nside=64,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        hp_map.data_1d["counts"] = xr.DataArray(
            data=np.fromfunction(
                lambda time, energy, pixel: 1000 + pixel * (10 * (energy + 1)),
                shape=(1, 10, hp_map.num_points),
                dtype=np.float32,
            ),
            dims=["epoch", "energy", "pixel"],
        )
        hp_map.data_1d["exposure_factor"] = xr.DataArray(
            data=np.ones((10, hp_map.num_points)),
            dims=["energy", "pixel"],
        )
        hp_map.data_1d["observation_date"] = xr.DataArray(
            data=np.ones(hp_map.num_points),
            dims=["pixel"],
        )

        rect_map, subdiv_depth_dict = hp_map.to_rectangular_skymap(
            rect_spacing_deg=2,
            value_keys=["counts", "exposure_factor", "observation_date"],
        )

        for value_key, subdiv_depth in subdiv_depth_dict.items():
            # subdiv depth should always be between 1 and
            # ena_maps.MAX_SUBDIV_RECURSION_DEPTH
            np.testing.assert_array_less(
                0,
                subdiv_depth,
                err_msg=f"subdiv <1 for: {value_key}",
            )
            np.testing.assert_array_less(
                subdiv_depth,
                ena_maps.MAX_SUBDIV_RECURSION_DEPTH + 1,
                err_msg=f"subdiv >MAX for: {value_key}",
            )

            # The min and max values of the rect and healpix maps should be close
            # The min will have a larger relative tolerance because the variation
            # in the test data is larger in comparison to the min value than to the max
            np.testing.assert_allclose(
                rect_map.data_1d[value_key].min(),
                hp_map.data_1d[value_key].min(),
                rtol=5e-2,
                err_msg=f"Min values of {value_key} do not match",
            )
            np.testing.assert_allclose(
                rect_map.data_1d[value_key].max(),
                hp_map.data_1d[value_key].max(),
                rtol=1e-3,
                err_msg=f"Max values of {value_key} do not match",
            )

            # The dims of the rect map should be the same as the healpix map,
            # except for the final pixel dimension
            assert (
                rect_map.data_1d[value_key].dims[:-1]
                == hp_map.data_1d[value_key].dims[:-1]
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
                l1c_product,
                spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
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
            self.rectangular_l1c_pset_products[0],
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
            self.rectangular_l1c_pset_products[0],
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
            self.rectangular_l1c_pset_products[0],
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
            self.rectangular_l1c_pset_products[0],
            spice_reference_frame=geometry.SpiceFrame.IMAP_DPS,
        )
        mock_pset_output_frame = ena_maps.RectangularPointingSet(
            self.rectangular_l1c_pset_products[1],
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


class TestAbstractSkyMap:
    @pytest.mark.parametrize(
        "skymap_props_dict",
        [
            pytest.param(
                # HealpixSkyMap properties
                {
                    "sky_tiling_type": "HEALPIX",
                    "nside": 32,
                    "nested": True,
                    "spice_reference_frame": geometry.SpiceFrame.ECLIPJ2000.name,
                    "values_to_push_project": ["foo", "bar"],
                },
                id="healpix-skymap",
            ),
            pytest.param(
                {
                    "sky_tiling_type": "RECTANGULAR",
                    "spacing_deg": 2,
                    "spice_reference_frame": geometry.SpiceFrame.ECLIPJ2000.name,
                    "values_to_pull_project": ["potato", "po-tah-to"],
                },
                id="rectangular-skymap",
            ),
        ],
    )
    def test_to_dict_and_from_dict(self, skymap_props_dict):
        """Test serialization to and from dictionary"""
        # Make a SkyMap from the original properties dict
        skymap_from_dict = ena_maps.AbstractSkyMap.from_dict(skymap_props_dict)

        # Use the SkyMap to create a new properties dict
        dict_from_skymap = skymap_from_dict.to_dict()

        assert (
            skymap_from_dict.spice_reference_frame
            == geometry.SpiceFrame[skymap_props_dict["spice_reference_frame"]]
        )

        if skymap_props_dict["sky_tiling_type"] == "HEALPIX":
            assert isinstance(skymap_from_dict, ena_maps.HealpixSkyMap), (
                "from_dict should return a HealpixSkyMap object"
            )
            assert skymap_from_dict.nside == skymap_props_dict["nside"]
            assert skymap_from_dict.nested == skymap_props_dict["nested"]
            assert (
                skymap_from_dict.values_to_push_project
                == skymap_props_dict["values_to_push_project"]
            )
            assert skymap_from_dict.values_to_pull_project == []

        elif skymap_props_dict["sky_tiling_type"] == "RECTANGULAR":
            assert isinstance(skymap_from_dict, ena_maps.RectangularSkyMap), (
                "from_dict should return a RectangularSkyMap object"
            )
            assert skymap_from_dict.spacing_deg == skymap_props_dict["spacing_deg"]
            assert skymap_from_dict.values_to_push_project == []
            assert (
                skymap_from_dict.values_to_pull_project
                == skymap_props_dict["values_to_pull_project"]
            )

        for key in [
            "sky_tiling_type",
            "spice_reference_frame",
            "nside",
            "nested",
            "spacing_deg",
        ]:
            if key in skymap_props_dict:
                assert dict_from_skymap[key] == skymap_props_dict[key]

        # Check that the dict from the SkyMap matches the original dict ONLY after
        # adding automatically added "values_to_push_project"/"values_to_pull_project"
        # key to the original dict
        assert dict_from_skymap != skymap_props_dict

        # In the dicts passed in above, the HEALPIX one is missing the pull key
        # and the RECTANGULAR one is missing the push key
        if skymap_props_dict["sky_tiling_type"] == "HEALPIX":
            skymap_props_dict["values_to_pull_project"] = []
        elif skymap_props_dict["sky_tiling_type"] == "RECTANGULAR":
            skymap_props_dict["values_to_push_project"] = []
        assert dict_from_skymap == skymap_props_dict

        # Change a value in the new dict and check that it is not equal to the original
        dict_from_skymap["spice_reference_frame"] = "SPACE!"
        assert (
            dict_from_skymap["spice_reference_frame"]
            != skymap_props_dict["spice_reference_frame"]
        )

    def test_to_json_and_from_json(self):
        """Test serialization to and from JSON"""
        # Make a SkyMap from the original properties dict
        skymap_props_dict = {
            "sky_tiling_type": "HEALPIX",
            "nside": 32,
            "nested": True,
            "spice_reference_frame": geometry.SpiceFrame.ECLIPJ2000.name,
            "values_to_push_project": ["foo", "bar"],
        }

        # Write a temporary json file with the properties dict

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", mode="w"
        ) as temp_file:
            json.dump(skymap_props_dict, temp_file)
            temp_file_path_input = temp_file.name

        # Read the json file and create a new SkyMap from it
        skymap_from_json = ena_maps.AbstractSkyMap.from_json(temp_file_path_input)

        # Create json output from the SkyMap at a separate temporary file path
        temp_file_path_output = tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", mode="w"
        ).name
        skymap_from_json.to_json(json_path=temp_file_path_output)

        assert skymap_from_json.spice_reference_frame == geometry.SpiceFrame.ECLIPJ2000
        assert skymap_from_json.tiling_type is ena_maps.SkyTilingType.HEALPIX
        assert skymap_from_json.nside == 32
        assert skymap_from_json.nested is True
        assert skymap_from_json.values_to_push_project == ["foo", "bar"]
        assert skymap_from_json.values_to_pull_project == []

        # Expect there to be a AttributeError when accessing a non-existent key
        with pytest.raises(AttributeError):
            _ = skymap_from_json.spacing_deg

        # Check that the json output is the same as the original input ONLY
        # after adding automatically added
        # "values_to_push_project"/"values_to_pull_project" key to the original dict
        with open(temp_file_path_input) as f:
            original_json = json.load(f)
        with open(temp_file_path_output) as f:
            output_json = json.load(f)
        # The output json will have added an empty list for values_to_pull_project
        assert original_json != output_json
        # add the values_to_pull_project key to the original json
        original_json["values_to_pull_project"] = []
        assert original_json == output_json
