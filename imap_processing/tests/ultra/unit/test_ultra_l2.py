from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy_healpix.healpy import nside2pixarea

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import write_cdf
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.quality_flags import ImapPSETUltraFlags
from imap_processing.tests.ultra.mock_data import mock_l1c_pset_product_healpix
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import get_energy_delta_minus_plus
from imap_processing.ultra.l2 import ultra_l2
from imap_processing.ultra.l2.ultra_l2 import (
    VARIABLES_TO_AVERAGE_OVER_COARSE_ENERGY_BINS,
    build_default_coarse_bin_edges,
    calculate_systematic_uncertainty,
)

ENERGY_BIN_EDGES_PATH = (
    imap_module_directory
    / "tests"
    / "ultra"
    / "data"
    / "l2"
    / "imap_ultra_l2-energy-bin-group-sizes_20250101_v000.csv"
)


class TestUltraL2:
    @pytest.fixture
    def _setup_spice_kernels_list(self, spice_test_data_path, furnish_kernels):
        self.required_kernel_names = [
            "naif0012.tls",
            "imap_science_120.tf",
            "imap_sclk_0000.tsc",
            "sim_1yr_imap_attitude.bc",
            "sim_1yr_imap_pointing_frame.bc",
        ]

    @pytest.fixture
    def _mock_single_pset(self, _setup_spice_kernels_list, furnish_kernels):
        with furnish_kernels(self.required_kernel_names):
            self.ultra_pset = mock_l1c_pset_product_healpix(
                nside=16,
                counts_nside=32,
                stripe_center_lat=0,
                timestr="2025-05-15T12:00:00",
                energy_dependent_exposure=True,
            )

    @pytest.fixture
    def _mock_multiple_psets(self, _setup_spice_kernels_list, furnish_kernels):
        # Set the timestrs to be 6 months apart from the 0th to final pset
        manual_timestrs = [
            "2025-05-15T12:00:00",
            "2025-07-15T12:00:00",
            "2025-09-15T12:00:00",
            "2025-11-15T12:00:00",
        ]

        with furnish_kernels(self.required_kernel_names):
            self.ultra_psets = [
                mock_l1c_pset_product_healpix(
                    nside=16,
                    counts_nside=32,
                    stripe_center_lat=mid_latitude,
                    width_scale=5,
                    counts_scaling_params=(50, 0.5),
                    peak_exposure=1000,
                    timestr=manual_timestrs[i],
                    head=("90"),
                    energy_dependent_exposure=True,
                )
                for i, mid_latitude in enumerate(
                    np.arange(
                        22.5,
                        180,
                        45,
                    )
                )
            ]
            # Add extra ultra specific variables to each pset
            for pset in self.ultra_psets:
                pset["efficiency"] = xr.ones_like(pset["sensitivity"])
                pset["scatter_theta"] = xr.ones_like(pset["sensitivity"])
                pset["scatter_phi"] = xr.ones_like(pset["sensitivity"])

        self.psets_total_counts = np.sum(
            [pset["counts"].values.sum() for pset in self.ultra_psets]
        )

    @pytest.fixture
    def mock_data_dict(self, _mock_multiple_psets):
        return {pset.attrs["Logical_file_id"]: pset for pset in self.ultra_psets}

    @pytest.mark.parametrize("epoch_dim_for_energy_delta", [True, False])
    @pytest.mark.parametrize(
        ["map_frame", "rtol"],
        [
            # Tight tolerance when 'projecting' to the same frame
            ("IMAP_DPS", 1e-8),
            # Loose tolerance of 30% error vs naive ena_intensity
            # estimate with real projection.
            # TODO: Ideally this tolerance will tighten if we can fix the issue with
            # the exposure time for uneven numbers of pixels from each PointingSet.
            ("ECLIPJ2000", 3e-1),
        ],
    )
    @pytest.mark.usefixtures("_mock_single_pset", "_setup_spice_kernels_list")
    def test_generate_ultra_rectangular_skymap_single_pset(
        self, epoch_dim_for_energy_delta, map_frame, rtol, furnish_kernels
    ):
        # Avoid modifying the original pset
        pset = mock_l1c_pset_product_healpix(
            nside=128,
            stripe_center_lat=0,
            timestr="2025-05-15T12:00:00",
            energy_dependent_exposure=True,
        )
        # Set the values in the single input PSET for easy calculation
        # of the expected ena_intensity and ena_intensity statistical uncertainty]
        counts_fillval = 10
        pset["counts"].values = np.full_like(pset["counts"].values, counts_fillval)
        pset["exposure_factor"].values = np.ones_like(pset["exposure_factor"])
        pset["background_rates"].values = np.ones_like(pset["background_rates"].values)
        pset["sensitivity"].values = np.ones_like(pset["sensitivity"].values)
        pset["geometric_function"].values = np.ones_like(pset["sensitivity"].values)
        pset["energy_bin_delta"].values = np.ones_like(pset["energy_bin_delta"].values)
        pset["efficiency"] = xr.ones_like(pset["sensitivity"])
        pset["scatter_theta"] = xr.ones_like(pset["sensitivity"])
        pset["scatter_phi"] = xr.ones_like(pset["sensitivity"])

        pset["energy_bin_delta"].values = np.ones_like(pset["energy_bin_delta"].values)
        if epoch_dim_for_energy_delta:
            # add an extra dim to the start
            pset["energy_bin_delta"] = pset["energy_bin_delta"].expand_dims(
                {CoordNames.TIME.value: pset["epoch"].values}
            )
        # Create the rectangular skymap in the desired frame.
        with furnish_kernels(self.required_kernel_names):
            rec_skymap, _ = ultra_l2.generate_ultra_skymap(
                ultra_l1c_psets=[
                    pset,
                ],
                output_map_structure=ena_maps.AbstractSkyMap.from_properties_dict(
                    {
                        "sky_tiling_type": "RECTANGULAR",
                        "spice_reference_frame": "ECLIPJ2000",
                        "values_to_push_project": [
                            "counts",
                        ],
                        "values_to_pull_project": [
                            "exposure_factor",
                            "sensitivity",
                            "geometric_function",
                            "efficiency",
                            "scatter_theta",
                            "scatter_phi",
                            "background_rates",
                        ],
                        "spacing_deg": 2.0,
                    }
                ),
                build_rectangular_map=True,
            )

        assert rec_skymap.spacing_deg == 2.0

        # Check that required variables are present, and dropped variables are not
        expected_vars = [
            "counts",
            "background_rates",
            "obs_date_range",
            "exposure_factor",
            "sensitivity",
            "geometric_function",
            "efficiency",
            "scatter_theta",
            "scatter_phi",
            "obs_date",
        ]
        for var in expected_vars:
            assert var in rec_skymap.data_1d.data_vars
        unexpected_vars = ultra_l2.VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION
        for var in unexpected_vars:
            assert var not in rec_skymap.data_1d.data_vars

    @pytest.mark.parametrize("epoch_dim_for_energy_delta", [True, False])
    @pytest.mark.parametrize(
        ["map_frame", "rtol"],
        [
            # Tight tolerance when 'projecting' to the same frame
            ("IMAP_DPS", 1e-8),
            # Loose tolerance of 30% error vs naive ena_intensity
            # estimate with real projection.
            # TODO: Ideally this tolerance will tighten if we can fix the issue with
            # the exposure time for uneven numbers of pixels from each PointingSet.
            ("ECLIPJ2000", 3e-1),
        ],
    )
    @pytest.mark.usefixtures("_mock_single_pset", "_setup_spice_kernels_list")
    def test_generate_ultra_healpix_skymap_single_pset(
        self, epoch_dim_for_energy_delta, map_frame, rtol, furnish_kernels
    ):
        # Avoid modifying the original pset
        pset = mock_l1c_pset_product_healpix(
            nside=128,
            stripe_center_lat=0,
            timestr="2025-05-15T12:00:00",
            energy_dependent_exposure=True,
        )
        # Set the values in the single input PSET for easy calculation
        # of the expected ena_intensity and ena_intensity statistical uncertainty
        counts_fillval = 10
        pset["counts"].values = np.full_like(pset["counts"].values, counts_fillval)
        pset["exposure_factor"].values = np.ones_like(pset["exposure_factor"])
        pset["background_rates"].values = np.ones_like(pset["background_rates"].values)
        pset["sensitivity"].values = np.ones_like(pset["sensitivity"].values)
        pset["geometric_function"].values = np.ones_like(pset["sensitivity"].values)
        pset["energy_bin_delta"].values = np.ones_like(pset["energy_bin_delta"].values)
        pset["efficiency"] = xr.ones_like(pset["sensitivity"])
        pset["scatter_theta"] = xr.ones_like(pset["sensitivity"])
        pset["scatter_phi"] = xr.ones_like(pset["sensitivity"])

        pset["energy_bin_delta"].values = np.ones_like(pset["energy_bin_delta"].values)
        if epoch_dim_for_energy_delta:
            # add an extra dim to the start
            pset["energy_bin_delta"] = pset["energy_bin_delta"].expand_dims(
                {CoordNames.TIME.value: pset["epoch"].values}
            )
        # Create the Healpix skymap in the desired frame.
        with furnish_kernels(self.required_kernel_names):
            hp_skymap, _ = ultra_l2.generate_ultra_skymap(
                ultra_l1c_psets=[
                    pset,
                ],
                output_map_structure=ena_maps.AbstractSkyMap.from_properties_dict(
                    {
                        "sky_tiling_type": "HEALPIX",
                        "spice_reference_frame": map_frame,
                        "values_to_push_project": [
                            "counts",
                        ],
                        "values_to_pull_project": [
                            "exposure_factor",
                            "sensitivity",
                            "geometric_function",
                            "efficiency",
                            "scatter_theta",
                            "scatter_phi",
                            "background_rates",
                        ],
                        "nside": 32,
                        "nested": False,
                    }
                ),
            )

        assert hp_skymap.nside == 32
        assert hp_skymap.nested is False

        # Check that required variables are present, and dropped variables are not
        expected_vars = [
            "counts",
            "background_rates",
            "ena_intensity",
            "obs_date_range",
            "ena_intensity_stat_uncert",
            "exposure_factor",
            "sensitivity",
            "geometric_function",
            "efficiency",
            "scatter_theta",
            "scatter_phi",
            "obs_date",
        ]
        for var in expected_vars:
            assert var in hp_skymap.data_1d.data_vars
        unexpected_vars = ultra_l2.VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION
        for var in unexpected_vars:
            assert var not in hp_skymap.data_1d.data_vars

        # The ratio of the solid angle of a map's pixel to the solid angle of a
        # pointing set's pixel. Counts will scale up by this ratio.
        solid_angle_ratio_map_to_pset = (
            hp_skymap.solid_angle / ena_maps.UltraPointingSet(pset).solid_angle
        )

        # Estimate the expected ena_intensity and its uncertainty
        expected_ena_intensity = (
            counts_fillval * solid_angle_ratio_map_to_pset / 1
        ) / (1 * hp_skymap.solid_angle * 1)
        # 4 fine bins per coarse bin
        binned_counts = (
            np.full(
                (len(hp_skymap.data_1d["energy_bin_geometric_mean"])), counts_fillval
            )
            * 4
        )
        # last bin only has 2 fine bins
        binned_counts[-1] = counts_fillval * 2
        energy_bin_deltas = np.full(12, 4)
        # The last delta is 2 instead of 4
        energy_bin_deltas[-1] = 2
        expected_ena_intensity_unc = (
            (binned_counts * solid_angle_ratio_map_to_pset) ** 0.5 / 1
        ) / (1 * hp_skymap.solid_angle * energy_bin_deltas)

        n_pixels = hp_skymap.data_1d["ena_intensity_stat_uncert"].shape[2]
        expected_ena_intensity_unc = np.tile(
            expected_ena_intensity_unc[np.newaxis, :, np.newaxis], (1, 1, n_pixels)
        )

        np.testing.assert_allclose(
            hp_skymap.data_1d["ena_intensity"].values,
            expected_ena_intensity,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            expected_ena_intensity_unc,
            hp_skymap.data_1d["ena_intensity_stat_uncert"].values,
            rtol=rtol,
        )

    @pytest.mark.parametrize("epoch_dim_for_energy_delta", [True, False])
    @pytest.mark.parametrize(
        ["map_frame", "rtol"],
        [
            # Tight tolerance when 'projecting' to the same frame
            ("IMAP_DPS", 1e-8),
            # Loose tolerance of 30% error vs naive ena_intensity
            # estimate with real projection.
            # TODO: Ideally this tolerance will tighten if we can fix the issue with
            # the exposure time for uneven numbers of pixels from each PointingSet.
            ("ECLIPJ2000", 3e-1),
        ],
    )
    @pytest.mark.usefixtures("_mock_single_pset", "_setup_spice_kernels_list")
    def test_generate_ultra_healpix_skymap_quality_flag(
        self, epoch_dim_for_energy_delta, map_frame, rtol, furnish_kernels
    ):
        # Avoid modifying the original pset
        pset = self.ultra_pset.copy(deep=True)

        # Set the values in the single input PSET for easy calculation
        # of the expected ena_intensity and ena_intensity statistical uncertainty
        pset["counts"].values = np.full_like(pset["counts"].values, 10)
        pset["exposure_factor"].values = np.ones_like(pset["exposure_factor"].values)
        pset["background_rates"].values = np.ones_like(pset["background_rates"].values)
        pset["sensitivity"].values = np.ones_like(pset["sensitivity"].values)
        pset["energy_bin_delta"].values = np.ones_like(pset["energy_bin_delta"].values)

        pset_quality = pset.copy(deep=True)
        # Flag every other pixel (e.g., even indices)
        pset_quality["quality_flags"][0, ::2] = ImapPSETUltraFlags.EARTH_FOV.value

        if epoch_dim_for_energy_delta:
            # add an extra dim to the start
            pset["energy_bin_delta"] = pset["energy_bin_delta"].expand_dims(
                {CoordNames.TIME.value: pset["epoch"].values}
            )

        # Create the Healpix skymap in the desired frame.
        with furnish_kernels(self.required_kernel_names):
            hp_skymap, _ = ultra_l2.generate_ultra_skymap(
                ultra_l1c_psets=[pset, pset_quality],
                output_map_structure=ena_maps.AbstractSkyMap.from_properties_dict(
                    {
                        "sky_tiling_type": "HEALPIX",
                        "spice_reference_frame": map_frame,
                        "values_to_push_project": [
                            "counts",
                        ],
                        "values_to_pull_project": [
                            "exposure_factor",
                            "sensitivity",
                            "background_rates",
                        ],
                        "nside": 32,
                        "nested": False,
                    }
                ),
            )

        assert hp_skymap.nside == 32
        assert hp_skymap.nested is False

        # Check that required variables are present, and dropped variables are not
        expected_vars = [
            "counts",
            "background_rates",
            "ena_intensity",
            "obs_date_range",
            "ena_intensity_stat_uncert",
            "exposure_factor",
            "obs_date",
        ]
        for var in expected_vars:
            assert var in hp_skymap.data_1d.data_vars
        unexpected_vars = ultra_l2.VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION
        for var in unexpected_vars:
            assert var not in hp_skymap.data_1d.data_vars

        energy_bins = 46  # Original number of fine energy bins
        n_pix = 12288
        n_counts = 10 * energy_bins * n_pix * 1.5

        # The total counts in the skymap should be equal to the sum of the counts
        # in the individual psets
        np.testing.assert_allclose(
            hp_skymap.data_1d["counts"].sum(),
            n_counts,
        )

    @pytest.mark.usefixtures("_mock_multiple_psets", "_setup_spice_kernels_list")
    def test_generate_ultra_healpix_skymap_multiple_psets(self, furnish_kernels):
        with patch(
            "imap_processing.ultra.l2.ultra_l2.VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION",
            [],
        ):
            with furnish_kernels(self.required_kernel_names):
                hp_skymap, pset_epochs = ultra_l2.generate_ultra_skymap(
                    ultra_l1c_psets=self.ultra_psets,
                    output_map_structure=ena_maps.AbstractSkyMap.from_properties_dict(
                        {
                            "sky_tiling_type": "RECTANGULAR",
                            "spice_reference_frame": "ECLIPJ2000",
                            "values_to_push_project": [
                                "counts",
                            ],
                            "values_to_pull_project": [
                                "exposure_factor",
                                "sensitivity",
                                "background_rates",
                            ],
                            "spacing_deg": 2.0,
                        }
                    ),
                )
        assert len(pset_epochs) == len(self.ultra_psets)

        assert hp_skymap.nside == ultra_l2.DEFAULT_L2_HEALPIX_NSIDE
        assert hp_skymap.nested == ultra_l2.DEFAULT_L2_HEALPIX_NESTED

        # The total counts in the skymap should be equal to the sum of the counts
        # in the individual psets
        np.testing.assert_allclose(
            hp_skymap.data_1d["counts"].sum(),
            self.psets_total_counts,
        )
        # The pointing independent variables should have been pulled once
        np.testing.assert_allclose(
            hp_skymap.data_1d["geometric_function"],
            np.ones_like(hp_skymap.data_1d["geometric_function"]),
        )
        # The map should contain the following variables,
        # because we did not drop any variables
        expected_vars = (
            ultra_l2.REQUIRED_L1C_VARIABLES_PUSH
            + ultra_l2.REQUIRED_L1C_VARIABLES_PULL
            + ultra_l2.VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION
            + ultra_l2.EXPECTED_L1C_POINTING_INDEPENDENT_VARIABLES_PULL
            + ["ena_intensity", "ena_intensity_stat_uncert"]
        )
        for var in expected_vars:
            assert var in hp_skymap.data_1d.data_vars

        # Check the dims of the key variables
        counts_dims = (
            CoordNames.TIME.value,
            CoordNames.ENERGY_ULTRA_L1C.value,
            CoordNames.GENERIC_PIXEL.value,
        )
        pointing_independent_dims = (
            CoordNames.ENERGY_ULTRA_L1C.value,
            CoordNames.GENERIC_PIXEL.value,
        )
        assert hp_skymap.data_1d["counts"].dims == counts_dims
        assert hp_skymap.data_1d["ena_intensity"].dims == counts_dims
        assert hp_skymap.data_1d["ena_intensity_stat_uncert"].dims == counts_dims
        assert hp_skymap.data_1d["exposure_factor"].dims == counts_dims
        assert hp_skymap.data_1d["background_rates"].dims == counts_dims

        assert hp_skymap.data_1d["sensitivity"].dims == pointing_independent_dims
        assert hp_skymap.data_1d["efficiency"].dims == pointing_independent_dims
        assert hp_skymap.data_1d["geometric_function"].dims == pointing_independent_dims
        assert hp_skymap.data_1d["scatter_theta"].dims == pointing_independent_dims
        assert hp_skymap.data_1d["scatter_phi"].dims == pointing_independent_dims

    @pytest.mark.external_test_data
    @pytest.mark.usefixtures("_setup_spice_kernels_list")
    def test_ultra_l2_output_unbinned_healpix(self, mock_data_dict, furnish_kernels):
        map_structure = ena_maps.AbstractSkyMap.from_properties_dict(
            {
                "sky_tiling_type": "HEALPIX",
                "spice_reference_frame": "ECLIPJ2000",
                "values_to_push_project": [
                    "counts",
                ],
                "values_to_pull_project": [
                    "exposure_factor",
                    "sensitivity",
                    "efficiency",
                    "sensitivity",
                    "geometric_function",
                    "scatter_theta",
                    "scatter_phi",
                ],
                "nside": 16,
                "nested": True,
            }
        )
        with furnish_kernels(self.required_kernel_names):
            [
                map_dataset,
            ] = ultra_l2.ultra_l2(
                data_dict=mock_data_dict,
                output_map_structure=map_structure,
                energy_bin_edges_file=ENERGY_BIN_EDGES_PATH,
            )

        assert (
            map_dataset.attrs["Logical_source"]
            == "imap_ultra_l2_u90-ena-h-unknown-nsp-full-hae-nside16-6mo"
        )
        assert "unknown frame" in map_dataset.attrs["Logical_source_description"]

        assert map_dataset.attrs["HEALPix_nside"] == str(map_structure.nside)
        assert map_dataset.attrs["HEALPix_nest"] == str(map_structure.nested)
        assert "6mo" in map_dataset.attrs["Logical_source"]

        assert (
            map_dataset["ena_intensity_sys_err"].shape
            == map_dataset["ena_intensity"].shape
        )
        expected_unc = np.broadcast_to(
            np.array(
                [
                    0.380,
                    0.347,
                    0.309,
                    0.282,
                    0.318,
                    0.348,
                    0.380,
                    0.380,
                    0.380,
                    0.380,
                    0.380,
                    0.380,
                ]
            )[np.newaxis, :, np.newaxis],
            map_dataset["ena_intensity_stat_uncert"].shape,
        )

        np.testing.assert_allclose(
            map_dataset["ena_intensity_sys_err"],
            expected_unc * map_dataset["ena_intensity"],
            rtol=0,
            atol=1e-12,
        )

        # Check solid angle values and metadata
        np.testing.assert_allclose(
            map_dataset["solid_angle"],
            nside2pixarea(
                map_structure.nside,
                degrees=False,
            ),
        )
        assert map_dataset["solid_angle"].attrs["UNITS"] == "sr"

        # Check that the positional uncertainty variables were renamed
        assert "positional_uncert_theta" in map_dataset
        assert "positional_uncert_phi" in map_dataset
        # Check that background_rates was renamed to bg_rate
        assert "bg_rate" in map_dataset
        # Check energy deltas
        assert "energy_delta_plus" in map_dataset
        assert "energy_delta_minus" in map_dataset
        # Check epoch deltas
        assert map_dataset["epoch"].attrs["DELTA_PLUS_VAR"] == "epoch_delta"
        assert map_dataset["epoch"].attrs["DELTA_MINUS_VAR"] == "epoch_delta_minus"
        assert "epoch_delta" in map_dataset
        assert "epoch_delta_minus" in map_dataset
        np.testing.assert_array_equal(map_dataset["epoch_delta_minus"].values, 0)

    @pytest.mark.external_test_data
    @pytest.mark.usefixtures("_setup_spice_kernels_list")
    def test_ultra_l2_rectangular(self, mock_data_dict, furnish_kernels):
        rect_map_structure = ena_maps.AbstractSkyMap.from_properties_dict(
            {
                "sky_tiling_type": "RECTANGULAR",
                "spice_reference_frame": "ECLIPJ2000",
                "projection_method_and_values": {
                    "values_to_push_project": [
                        "counts",
                        "exposure_factor",
                        "sensitivity",
                    ],
                },
                "spacing_deg": 10,  # Larger spacing for faster test
            }
        )
        hp_map_structure = ena_maps.AbstractSkyMap.from_properties_dict(
            {
                "sky_tiling_type": "HEALPIX",
                "spice_reference_frame": "ECLIPJ2000",
                "projection_method_and_values": {
                    "values_to_push_project": [
                        "counts",
                        "exposure_factor",
                        "sensitivity",
                    ],
                },
                "nside": 16,
                "nested": True,
            }
        )
        # Create both a rectangular map and a healpix map
        with furnish_kernels(self.required_kernel_names):
            [
                rect_map_dataset,
            ] = ultra_l2.ultra_l2(
                data_dict=mock_data_dict,
                output_map_structure=rect_map_structure,
                energy_bin_edges_file=ENERGY_BIN_EDGES_PATH,
                store_subdivision_depth=False,
            )
            [
                hp_map_dataset,
            ] = ultra_l2.ultra_l2(
                data_dict=mock_data_dict,
                energy_bin_edges_file=ENERGY_BIN_EDGES_PATH,
                output_map_structure=hp_map_structure,
            )

        assert rect_map_dataset.attrs["Spacing_degrees"] == str(
            rect_map_structure.spacing_deg
        )
        assert (
            rect_map_dataset.attrs["Spice_reference_frame"]
            == rect_map_structure.spice_reference_frame.name
        )

        # Check the dims of the key variables
        expected_ena_intensity_dims = (
            CoordNames.TIME.value,
            CoordNames.ENERGY_L2.value,
            CoordNames.AZIMUTH_L2.value,
            CoordNames.ELEVATION_L2.value,
        )
        assert rect_map_dataset["ena_intensity"].dims == expected_ena_intensity_dims
        assert (
            rect_map_dataset["ena_intensity_stat_uncert"].dims
            == expected_ena_intensity_dims
        )
        assert rect_map_dataset["exposure_factor"].dims == expected_ena_intensity_dims

        # Check that '_label' coordinates were added for all coordinates except 'epoch'
        for coord_var in expected_ena_intensity_dims[1:]:
            assert f"{coord_var}_label" in rect_map_dataset.coords

        # Check that '_subdivision_depth' variables were not added because the
        # store_subdivision_depth flag was set to False
        assert "ena_intensity_subdivision_depth" not in rect_map_dataset.data_vars

        # The mean ena_intensity should be close between the healpix / rectangular maps
        # Test they agree to within 1% of one another
        np.testing.assert_allclose(
            rect_map_dataset["ena_intensity"].mean(),
            hp_map_dataset["ena_intensity"].mean(),
            rtol=3e-1,
            atol=1e-12,
        )
        # Check energy deltas
        assert "energy_delta_plus" in rect_map_dataset
        assert "energy_delta_minus" in rect_map_dataset

        # Variable Metadata spot checks
        ena_intensity_attrs = rect_map_dataset["ena_intensity"].attrs
        assert ena_intensity_attrs["VAR_TYPE"] == "data"
        for depend_num, depend in enumerate(expected_ena_intensity_dims):
            assert ena_intensity_attrs[f"DEPEND_{depend_num}"] == depend
            if depend_num > 0:
                assert (
                    ena_intensity_attrs[f"LABL_PTR_{depend_num}"] == f"{depend}_label"
                )
        assert ena_intensity_attrs["UNITS"] == "cm -2 s -1 sr -1 keV -1"

        exposure_attrs = rect_map_dataset["exposure_factor"].attrs
        assert exposure_attrs["VAR_TYPE"] == "data"
        assert exposure_attrs["UNITS"] == "s"

    @pytest.mark.parametrize(
        "tiling",
        [
            "healpix",
            "rectangular",
        ],
    )
    def test_get_variable_attributes_optional_energy_dependence(self, tiling):
        # Setup CDF attributes class
        cdf_attrs = ImapCdfAttributes()
        cdf_attrs.add_instrument_variable_attrs(instrument="enamaps", level="l2-common")
        cdf_attrs.add_instrument_variable_attrs(
            instrument="enamaps", level=f"l2-{tiling}"
        )

        exposure_factor_array_energy_independent = xr.DataArray(
            name="exposure_factor",
            data=np.ones((1, 16)),
            dims=(
                CoordNames.TIME.value,
                CoordNames.GENERIC_PIXEL.value,
            ),
        )
        exposure_factor_array_energy_dependent = xr.DataArray(
            name="exposure_factor",
            data=np.ones((1, 24, 16)),
            dims=(
                CoordNames.TIME.value,
                CoordNames.ENERGY_L2.value,
                CoordNames.GENERIC_PIXEL.value,
            ),
        )

        # Check the energy independent case attributes
        attrs_with_energy_independent_exposure = (
            ultra_l2.get_variable_attributes_optional_energy_dependence(
                cdf_attrs=cdf_attrs,
                variable_array=exposure_factor_array_energy_independent,
            )
        )

        # Check non dimensioned attributes
        assert attrs_with_energy_independent_exposure["UNITS"] == "s"

        # Check the depends (dimensions)
        assert (
            attrs_with_energy_independent_exposure["DEPEND_0"] == CoordNames.TIME.value
        )
        if tiling == "healpix":
            assert (
                attrs_with_energy_independent_exposure["DEPEND_1"]
                == CoordNames.HEALPIX_INDEX.value
            )
        elif tiling == "rectangular":
            assert (
                attrs_with_energy_independent_exposure["DEPEND_1"]
                == CoordNames.AZIMUTH_L2.value
            )
            assert (
                attrs_with_energy_independent_exposure["DEPEND_2"]
                == CoordNames.ELEVATION_L2.value
            )

        # Check the energy dependent case attributes
        attrs_with_energy_dependent_exposure = (
            ultra_l2.get_variable_attributes_optional_energy_dependence(
                cdf_attrs=cdf_attrs,
                variable_array=exposure_factor_array_energy_dependent,
            )
        )

        # Check an un-altered attribute (set by energy independent metadata and not
        # modified by energy dependent metadata)
        assert attrs_with_energy_dependent_exposure["UNITS"] == "s"

        # Check the depends (dimensions)
        assert attrs_with_energy_dependent_exposure["DEPEND_0"] == CoordNames.TIME.value
        assert (
            attrs_with_energy_dependent_exposure["DEPEND_1"]
            == CoordNames.ENERGY_L2.value
        )
        if tiling == "healpix":
            assert (
                attrs_with_energy_dependent_exposure["DEPEND_2"]
                == CoordNames.HEALPIX_INDEX.value
            )
        elif tiling == "rectangular":
            assert (
                attrs_with_energy_dependent_exposure["DEPEND_2"]
                == CoordNames.AZIMUTH_L2.value
            )
            assert (
                attrs_with_energy_dependent_exposure["DEPEND_3"]
                == CoordNames.ELEVATION_L2.value
            )

    @pytest.mark.usefixtures("_setup_spice_kernels_list")
    def test_ultra_l2_error_for_push_and_pull(
        self, mock_data_dict, furnish_kernels, caplog
    ):
        map_structure = ena_maps.AbstractSkyMap.from_properties_dict(
            {
                "sky_tiling_type": "HEALPIX",
                "spice_reference_frame": "ECLIPJ2000",
                "values_to_push_project": ["counts", "exposure_factor"],
                "values_to_pull_project": ["exposure_factor", "sensitivity"],
                "nside": 16,
                "nested": True,
            }
        )
        # An error is expected when the same variable is in both the push/pull lists
        with furnish_kernels(self.required_kernel_names):
            with pytest.raises(ValueError, match="Some variables are present in both"):
                ultra_l2.ultra_l2(
                    data_dict=mock_data_dict,
                    output_map_structure=map_structure,
                )

    def test_ultra_l2_descriptor_rectmap(self, mock_data_dict, furnish_kernels):
        with furnish_kernels(self.required_kernel_names):
            output_map = ultra_l2.ultra_l2(
                data_dict=mock_data_dict,
                descriptor="u90-ena-h-hf-nsp-full-hae-6deg-6mo",
            )[0]

        assert (
            output_map.attrs["Logical_source"]
            == "imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-6deg-6mo"
        )
        assert "heliospheric frame" in output_map.attrs["Logical_source_description"]

        assert output_map.attrs["Spice_reference_frame"] == "IMAP_HAE"
        assert output_map.attrs["Spacing_degrees"] == "6.0"
        # Variable Metadata spot checks
        assert (
            output_map["ena_intensity"].attrs["CATDESC"]
            == "IMAP Ultra90 H Inten, HAE Helio Frame, No Surv Corr, Full Spin,"
            " 6 deg, 6 Mon"
        )
        assert (
            output_map["counts"].attrs["CATDESC"]
            == "IMAP Ultra90 H Counts, HAE Helio Frame, No Surv Corr, Full Spin,"
            " 6 deg, 6 Mon"
        )
        write_cdf(output_map)

    @pytest.mark.usefixtures("_setup_spice_kernels_list")
    def test_ultra_l2_descriptor_hpmap(self, mock_data_dict, furnish_kernels):
        with furnish_kernels(self.required_kernel_names):
            output_map = ultra_l2.ultra_l2(
                data_dict=mock_data_dict,
                descriptor="u90-ena-h-sf-nsp-full-hae-nside32-3mo",
            )[0]

        assert "spacecraft frame" in output_map.attrs["Logical_source_description"]
        # Check that the logical source contains the expected information from the
        # descriptor string
        assert (
            output_map.attrs["Logical_source"]
            == "imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-nside32-3mo"
        )
        assert output_map.attrs["Spice_reference_frame"] == "IMAP_HAE"
        assert output_map.attrs["HEALPix_nside"] == "32"

        write_cdf(output_map)

    @pytest.mark.usefixtures("_mock_single_pset")
    def test_bin_pset_energy_bins_default(self):
        """Test binning with default bin sizes."""
        pset = mock_l1c_pset_product_healpix(
            nside=16,
            counts_nside=16,
            stripe_center_lat=0,
            timestr="2025-05-15T12:00:00",
            energy_dependent_exposure=True,
        )
        # Set the values in the single input PSET
        # Create a mock array with known values to test binning
        # e.g., 0,0,0,0,1,1,1,1,2,2,2,2,...11,11
        n_fine_bins = pset.energy_bin_geometric_mean.size
        n_coarse_bins = len(build_default_coarse_bin_edges()) - 1
        mock_vals = np.repeat(np.arange(n_coarse_bins), 4)[0:n_fine_bins]
        mock_array = (
            np.ones_like(pset["exposure_factor"]) * mock_vals[np.newaxis, :, np.newaxis]
        )
        pset["counts"].values = (
            np.ones_like(pset["counts"]) * mock_vals[np.newaxis, :, np.newaxis]
        )
        pset["exposure_factor"].values = mock_array
        pset["sensitivity"].values = mock_array[0]
        pset["geometric_function"].values = mock_array[0]
        pset["efficiency"].values = mock_array[0]
        pset["background_rates"].values = mock_array
        pset["scatter_theta"].values = mock_array[0]
        pset["scatter_phi"].values = mock_array[0]
        pset["energy_bin_delta"].values = np.ones_like(pset["energy_bin_delta"])
        # Bin the pset
        binned_pset = ultra_l2.bin_pset_energy_bins(
            pset, build_default_coarse_bin_edges()
        )
        # Check that the new bin edges are as expected
        expected_bin_edges = np.array(
            [
                3.0,
                4.6,
                6.96,
                10.27,
                15.71,
                23.4444,
                34.9866,
                52.2113,
                77.9161,
                116.276,
                173.521,
                258.95,
                316.335,
            ]
        )
        expected_energy_delta_minus, expected_energy_delta_plus = (
            get_energy_delta_minus_plus(expected_bin_edges)
        )
        np.testing.assert_array_equal(
            binned_pset["energy_delta_plus"].values,
            expected_energy_delta_plus,
        )
        np.testing.assert_array_equal(
            binned_pset["energy_delta_minus"].values,
            expected_energy_delta_minus,
        )
        # check that the pinned_pset energy_bin_geometric_mean values have been
        # recalculated correctly with the new bin edges
        np.testing.assert_array_equal(
            binned_pset["energy_bin_geometric_mean"].values,
            np.sqrt(expected_bin_edges[:-1] * expected_bin_edges[1:]),
        )

        # Check that the counts have been summed correctly in the new bins
        # 4 fine bins per coarse bin
        expected_binned_counts = np.arange(n_coarse_bins) * 4
        # The last bin only has 2 fine bins
        expected_binned_counts[-1] = (n_coarse_bins - 1) * 2
        # Broadcast to shape (1, n_coarse_bins, n_pixels)
        n_pixels = binned_pset["counts"].shape[2]
        expected_binned_counts = np.tile(
            expected_binned_counts[np.newaxis, :, np.newaxis], (1, 1, n_pixels)
        )
        np.testing.assert_array_equal(binned_pset["counts"], expected_binned_counts)

        # Check that the variables that should be averaged over the new bins are
        # correct
        for var in VARIABLES_TO_AVERAGE_OVER_COARSE_ENERGY_BINS:
            num_fine_bins_per_coarse_bin = np.full(n_coarse_bins, 4)
            # The last bin only has 2 fine bins
            num_fine_bins_per_coarse_bin[-1] = 2
            expected_binned_vals = (
                expected_binned_counts / num_fine_bins_per_coarse_bin[:, np.newaxis]
            )
            np.testing.assert_array_equal(
                binned_pset[var].squeeze(), expected_binned_vals[0]
            )

    @pytest.mark.usefixtures("_mock_single_pset")
    def test_bin_pset_energy_irregular_bins(self):
        """Test binning with irregular bin sizes."""
        # Avoid modifying the original pset
        pset = self.ultra_pset.copy(deep=True)
        pset["counts"].values = np.full(pset["counts"].shape, 2)
        # Update Exposure factor to be a ones array.
        pset["exposure_factor"].values = np.ones_like(pset["exposure_factor"])
        pset["background_rates"].values = np.ones_like(pset["background_rates"])
        # First group has 10 fine bins, then 30, then 6
        new_bin_edges = np.array([0, 10, 40, 46])
        # Bin the pset
        binned_pset = ultra_l2.bin_pset_energy_bins(pset, new_bin_edges)
        # Check that the new bin edges are as expected
        expected_bin_edges = np.array(
            [
                3.0,
                8.615,
                173.521,
                316.335,
            ]
        )
        expected_energy_delta_minus, expected_energy_delta_plus = (
            get_energy_delta_minus_plus(expected_bin_edges)
        )
        np.testing.assert_array_equal(
            binned_pset["energy_delta_plus"].values,
            expected_energy_delta_plus,
        )
        np.testing.assert_array_equal(
            binned_pset["energy_delta_minus"].values,
            expected_energy_delta_minus,
        )
        # check that the pinned_pset energy_bin_geometric_mean values have been
        # recalculated correctly with the new bin edges
        np.testing.assert_array_equal(
            binned_pset["energy_bin_geometric_mean"].values,
            np.sqrt(expected_bin_edges[:-1] * expected_bin_edges[1:]),
        )
        # Check that the counts have been summed correctly in the bins
        for i, group_size in enumerate([10, 30, 6]):
            counts_in_bin = binned_pset["counts"][0, i, :]
            expected_counts = np.full(
                counts_in_bin.shape,
                group_size * 2,
            )
            np.testing.assert_array_equal(counts_in_bin, expected_counts)

        # Check that the variables that should be averaged over the new bins are
        # correct
        # The arrays in the pset pre-binning are all be ones so they should
        # remain ones after averaging.
        for var in VARIABLES_TO_AVERAGE_OVER_COARSE_ENERGY_BINS:
            np.testing.assert_array_equal(np.unique(pset[var]), 1)

    @pytest.mark.usefixtures("_mock_single_pset")
    def test_bin_pset_energy_bin_subset(self):
        """Test binning with bins that do not include all of the fine bins."""
        # Avoid modifying the original pset
        pset = self.ultra_pset.copy(deep=True)
        # The bin edges do not cover all fine bins
        # In this case, we expect the function to ignore the fine bins outside
        # the new bin edges
        new_bin_edges = np.array([3, 10, 40])
        # Bin the pset
        binned_pset = ultra_l2.bin_pset_energy_bins(pset, new_bin_edges)
        # Check that the new bin edges are as expected
        expected_bin_edges = np.array(
            [
                4.2,
                8.615,
                173.521,
            ]
        )
        expected_energy_delta_minus, expected_energy_delta_plus = (
            get_energy_delta_minus_plus(expected_bin_edges)
        )
        np.testing.assert_array_equal(
            binned_pset["energy_delta_plus"].values,
            expected_energy_delta_plus,
        )
        np.testing.assert_array_equal(
            binned_pset["energy_delta_minus"].values,
            expected_energy_delta_minus,
        )
        # check that the pinned_pset energy_bin_geometric_mean values have been
        # recalculated correctly with the new bin edges
        np.testing.assert_array_equal(
            binned_pset["energy_bin_geometric_mean"].values,
            np.sqrt(expected_bin_edges[:-1] * expected_bin_edges[1:]),
        )
        # Check that the shapes are correct
        assert binned_pset["counts"].shape[1] == 2

        for var in VARIABLES_TO_AVERAGE_OVER_COARSE_ENERGY_BINS:
            assert binned_pset[var].squeeze().shape[0] == 2

        with pytest.raises(
            ValueError,
            match="The given bin_groups contain an index larger than the number of fine"
            " energy bins in the pset: 46.",
        ):
            # Bin the pset - should raise ValueError because the last bin edge is
            # out of range
            ultra_l2.bin_pset_energy_bins(pset, np.array([0, 10, 48]))

    @pytest.mark.usefixtures("_mock_single_pset")
    def test_bin_pset_energy_bins_zero_count_fine_bin(self):
        """Test binning with some fine bins having zero values."""
        # Avoid modifying the original pset
        pset = self.ultra_pset.copy(deep=True)
        # Set values in energy bin 0 to zero
        pset["sensitivity"][0, :].values = np.ones(pset["sensitivity"][0, :].shape)
        # Bin the pset
        binned_pset = ultra_l2.bin_pset_energy_bins(
            pset, build_default_coarse_bin_edges()
        )
        # Assert that the binned and averaged sensitivity in the first coarse bin is
        # equal to the average of the fine bins that were included (which excludes the
        # zero count bin)
        np.testing.assert_array_equal(
            binned_pset["sensitivity"].values, np.ones_like(binned_pset["sensitivity"])
        )


def test_calculate_systematic_uncertainty():
    """Test that the systematic uncertainty is calculated correctly."""
    sys_uncert_df = pd.read_csv(
        f"{imap_module_directory}/ultra/l2/ultra_l2_systematic_uncertainties.csv"
    )
    df = sys_uncert_df[sys_uncert_df["fm"] == 45]
    unc_array = calculate_systematic_uncertainty(45, df["energy"].values)

    assert unc_array.dims == ("energy",)
    np.testing.assert_array_equal(
        df["systematic_uncertainty"].values,
        unc_array,
    )


def test_calculate_systematic_uncertainty_wrong_energies():
    """Test that a value error is raised if the energies are unexpected"""
    with pytest.raises(
        ValueError,
        match="The energy values from the systematic uncertainty csv do not match the"
        " energy values of the map",
    ):
        calculate_systematic_uncertainty(90, np.arange(12))
