from unittest.mock import patch

import numpy as np
import pytest

from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.tests.ultra.data.mock_data import mock_l1c_pset_product_healpix
from imap_processing.ultra.l2 import ultra_l2


class TestUltraL2:
    @pytest.fixture
    def _setup_spice_kernels_list(self, spice_test_data_path, furnish_kernels):
        self.required_kernel_names = [
            "imap_science_0001.tf",
            "imap_sclk_0000.tsc",
            "sim_1yr_imap_attitude.bc",
            "sim_1yr_imap_pointing_frame.bc",
        ]

    @pytest.fixture
    def _mock_single_pset(self, _setup_spice_kernels_list, furnish_kernels):
        with furnish_kernels(self.required_kernel_names):
            self.ultra_pset = mock_l1c_pset_product_healpix(
                nside=128, stripe_center_lat=0, timestr="2025-05-15T12:00:00"
            )

    @pytest.fixture
    def _mock_multiple_psets(self, _setup_spice_kernels_list, furnish_kernels):
        with furnish_kernels(self.required_kernel_names):
            self.ultra_psets = [
                mock_l1c_pset_product_healpix(
                    nside=128,
                    stripe_center_lat=mid_latitude,
                    width_scale=5,
                    counts_scaling_params=(50, 0.5),
                    peak_exposure=1000,
                    timestr=f"2025-05-{4 * i + 1:02d}T12:00:00",
                    head=("90"),
                )
                for i, mid_latitude in enumerate(
                    np.arange(
                        22.5,
                        180,
                        45,
                    )
                )
            ]

        self.psets_total_counts = np.sum(
            [pset["counts"].values.sum() for pset in self.ultra_psets]
        )

    @pytest.fixture
    def mock_data_dict(self, _mock_multiple_psets):
        return {pset.attrs["Logical_file_id"]: pset for pset in self.ultra_psets}

    @pytest.mark.parametrize(
        ["map_frame", "rtol"],
        [
            # Tight tolerance when 'projecting' to the same frame
            ("IMAP_DPS", 1e-8),
            # Loose tolerance of  30% error vs naive flux estimate with real projection.
            # TODO: Ideally this tolerance will tighten if we can fix the issue with
            # the exposure time for uneven numbers of pixels from each PointingSet.
            ("ECLIPJ2000", 3e-1),
        ],
    )
    @pytest.mark.usefixtures("_mock_single_pset", "_setup_spice_kernels_list")
    def test_generate_ultra_healpix_skymap_single_pset(
        self, map_frame, rtol, furnish_kernels
    ):
        # Avoid modifying the original pset
        pset = self.ultra_pset.copy(deep=True)

        # Set the values in the single input PSET for easy calculation
        # of the expected flux and flux uncertainty
        pset["counts"].values = np.full_like(pset["counts"].values, 10)
        pset["exposure_factor"].values = np.ones_like(pset["exposure_factor"].values)
        pset["background_rates"].values = np.ones_like(pset["background_rates"].values)
        pset["sensitivity"].values = np.ones_like(pset["sensitivity"].values)
        pset["energy_bin_delta"].values = np.ones_like(pset["energy_bin_delta"].values)

        # Create the Healpix skymap in the desired frame.
        with furnish_kernels(self.required_kernel_names):
            hp_skymap = ultra_l2.generate_ultra_healpix_skymap(
                ultra_l1c_psets=[
                    pset,
                ],
                output_map_structure=ena_maps.AbstractSkyMap.from_dict(
                    {
                        "sky_tiling_type": "HEALPIX",
                        "spice_reference_frame": map_frame,
                        "projection_method_and_values": {
                            "PUSH": [
                                "counts",
                                "exposure_factor",
                                "sensitivity",
                                "background_rates",
                            ],
                        },
                        "nside": 32,
                        "nested": False,
                    }
                ),
            )

        assert hp_skymap.nside == 32
        assert hp_skymap.nested is False

        # Check that required variables are present, and dropped variables are not
        expected_vars = [
            "flux",
            "flux_uncertainty",
            "exposure_factor",
            "observation_time",
        ]
        for var in expected_vars:
            assert var in hp_skymap.data_1d.data_vars
        unexpected_vars = ultra_l2.VARIABLES_TO_DROP_AFTER_FLUX_CALCULATION
        for var in unexpected_vars:
            assert var not in hp_skymap.data_1d.data_vars

        # The ratio of the solid angle of a map's pixel to the solid angle of a
        # pointing set's pixel. Counts will scale up by this ratio.
        solid_angle_ratio_map_to_pset = (
            hp_skymap.solid_angle / ena_maps.UltraPointingSet(pset).solid_angle
        )

        # Estimate the expected flux and flux uncertainty
        expected_flux = (
            (10 * solid_angle_ratio_map_to_pset / 1) - 1 * solid_angle_ratio_map_to_pset
        ) / (1 * hp_skymap.solid_angle * 1)
        expected_flux_unc = ((10 * solid_angle_ratio_map_to_pset) ** 0.5 / 1) / (
            1 * hp_skymap.solid_angle * 1
        )

        np.testing.assert_allclose(
            hp_skymap.data_1d["flux"].values,
            expected_flux,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            hp_skymap.data_1d["flux_uncertainty"].values,
            expected_flux_unc,
            rtol=rtol,
        )

    @pytest.mark.usefixtures("_mock_multiple_psets", "_setup_spice_kernels_list")
    def test_generate_ultra_healpix_skymap_multiple_psets(self, furnish_kernels):
        with patch(
            "imap_processing.ultra.l2.ultra_l2.VARIABLES_TO_DROP_AFTER_FLUX_CALCULATION",
            [],
        ):
            with furnish_kernels(self.required_kernel_names):
                hp_skymap = ultra_l2.generate_ultra_healpix_skymap(
                    ultra_l1c_psets=self.ultra_psets,
                    output_map_structure=ena_maps.AbstractSkyMap.from_dict(
                        {
                            "sky_tiling_type": "RECTANGULAR",
                            "spice_reference_frame": "ECLIPJ2000",
                            "projection_method_and_values": {
                                "PUSH": [
                                    "counts",
                                    "exposure_factor",
                                    "sensitivity",
                                    "background_rates",
                                ],
                            },
                            "spacing_deg": 2.0,
                        }
                    ),
                )

        assert hp_skymap.nside == ultra_l2.DEFAULT_L2_HEALPIX_NSIDE
        assert hp_skymap.nested == ultra_l2.DEFAULT_L2_HEALPIX_NESTED

        # The total counts in the skymap should be equal to the sum of the counts
        # in the individual psets
        np.testing.assert_allclose(
            hp_skymap.data_1d["counts"].sum(),
            self.psets_total_counts,
        )

        # The map should contain the following variables,
        # because we did not drop any variables
        expected_vars = (
            ultra_l2.REQUIRED_L1C_VARIABLES
            + ultra_l2.VARIABLES_TO_DROP_AFTER_FLUX_CALCULATION
            + ["flux", "flux_uncertainty"]
        )
        for var in expected_vars:
            assert var in hp_skymap.data_1d.data_vars

        # Check the dims of the key variables
        counts_dims = (
            CoordNames.TIME.value,
            CoordNames.ENERGY_ULTRA.value,
            CoordNames.GENERIC_PIXEL.value,
        )
        assert hp_skymap.data_1d["counts"].dims == counts_dims
        assert hp_skymap.data_1d["flux"].dims == counts_dims
        assert hp_skymap.data_1d["flux_uncertainty"].dims == counts_dims
        assert hp_skymap.data_1d["exposure_factor"].dims == counts_dims[-1:]

    @pytest.mark.usefixtures("_setup_spice_kernels_list")
    def test_ultra_l2_output_unbinned_healpix(self, mock_data_dict, furnish_kernels):
        map_structure = ena_maps.AbstractSkyMap.from_dict(
            {
                "sky_tiling_type": "HEALPIX",
                "spice_reference_frame": "ECLIPJ2000",
                "projection_method_and_values": {
                    "PUSH": ["counts", "exposure_factor", "sensitivity"],
                },
                "nside": 16,
                "nested": True,
            }
        )
        with furnish_kernels(self.required_kernel_names):
            [
                map_dataset,
            ] = ultra_l2.ultra_l2(
                data_dict=mock_data_dict,
                data_version="001",
                output_map_structure=map_structure,
            )

        assert map_dataset.attrs["HEALPix_nside"] == map_structure.nside
        assert map_dataset.attrs["HEALPix_nest"] == map_structure.nested
