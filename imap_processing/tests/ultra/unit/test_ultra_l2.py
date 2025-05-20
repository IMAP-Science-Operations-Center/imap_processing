from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.tests.ultra.mock_data import mock_l1c_pset_product_healpix
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
                    nside=128,
                    stripe_center_lat=mid_latitude,
                    width_scale=5,
                    counts_scaling_params=(50, 0.5),
                    peak_exposure=1000,
                    timestr=manual_timestrs[i],
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
            # Loose tolerance of 30% error vs naive ena_intensity
            # estimate with real projection.
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
        # of the expected ena_intensity and ena_intensity statistical uncertainty
        pset["counts"].values = np.full_like(pset["counts"].values, 10)
        pset["exposure_factor"].values = np.ones_like(pset["exposure_factor"].values)
        pset["background_rates"].values = np.ones_like(pset["background_rates"].values)
        pset["sensitivity"].values = np.ones_like(pset["sensitivity"].values)
        pset["energy_bin_delta"].values = np.ones_like(pset["energy_bin_delta"].values)

        # Create the Healpix skymap in the desired frame.
        with furnish_kernels(self.required_kernel_names):
            hp_skymap, _ = ultra_l2.generate_ultra_healpix_skymap(
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
            "ena_intensity",
            "ena_intensity_stat_unc",
            "exposure_factor",
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
            (10 * solid_angle_ratio_map_to_pset / 1) - 1 * solid_angle_ratio_map_to_pset
        ) / (1 * hp_skymap.solid_angle * 1)
        expected_ena_intensity_unc = (
            (10 * solid_angle_ratio_map_to_pset) ** 0.5 / 1
        ) / (1 * hp_skymap.solid_angle * 1)

        np.testing.assert_allclose(
            hp_skymap.data_1d["ena_intensity"].values,
            expected_ena_intensity,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            hp_skymap.data_1d["ena_intensity_stat_unc"].values,
            expected_ena_intensity_unc,
            rtol=rtol,
        )

    @pytest.mark.usefixtures("_mock_multiple_psets", "_setup_spice_kernels_list")
    def test_generate_ultra_healpix_skymap_multiple_psets(self, furnish_kernels):
        with patch(
            "imap_processing.ultra.l2.ultra_l2.VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION",
            [],
        ):
            with furnish_kernels(self.required_kernel_names):
                hp_skymap, pset_epochs = ultra_l2.generate_ultra_healpix_skymap(
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

        # The map should contain the following variables,
        # because we did not drop any variables
        expected_vars = (
            ultra_l2.REQUIRED_L1C_VARIABLES_PUSH
            + ultra_l2.REQUIRED_L1C_VARIABLES_PULL
            + ultra_l2.VARIABLES_TO_DROP_AFTER_INTENSITY_CALCULATION
            + ["ena_intensity", "ena_intensity_stat_unc"]
        )
        for var in expected_vars:
            assert var in hp_skymap.data_1d.data_vars

        # Check the dims of the key variables
        counts_dims = (
            CoordNames.TIME.value,
            CoordNames.ENERGY_ULTRA_L1C.value,
            CoordNames.GENERIC_PIXEL.value,
        )
        assert hp_skymap.data_1d["counts"].dims == counts_dims
        assert hp_skymap.data_1d["ena_intensity"].dims == counts_dims
        assert hp_skymap.data_1d["ena_intensity_stat_unc"].dims == counts_dims
        assert hp_skymap.data_1d["exposure_factor"].dims == (
            CoordNames.TIME.value,
            CoordNames.GENERIC_PIXEL.value,
        )

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
            )

        assert map_dataset.attrs["HEALPix_nside"] == str(map_structure.nside)
        assert map_dataset.attrs["HEALPix_nest"] == str(map_structure.nested)
        assert "6mo" in map_dataset.attrs["Logical_source"]

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
                store_subdivision_depth=False,
            )
            [
                hp_map_dataset,
            ] = ultra_l2.ultra_l2(
                data_dict=mock_data_dict,
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
            rect_map_dataset["ena_intensity_stat_unc"].dims
            == expected_ena_intensity_dims
        )
        assert rect_map_dataset["exposure_factor"].dims == (
            CoordNames.TIME.value,
            CoordNames.AZIMUTH_L2.value,
            CoordNames.ELEVATION_L2.value,
        )

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
            rtol=1e-2,
            atol=1e-12,
        )

        # Variable Metadata spot checks
        ena_intensity_attrs = rect_map_dataset["ena_intensity"].attrs
        assert ena_intensity_attrs["VAR_TYPE"] == "data"
        for depend_num, depend in enumerate(expected_ena_intensity_dims):
            assert ena_intensity_attrs[f"DEPEND_{depend_num}"] == depend
            if depend_num > 0:
                assert (
                    ena_intensity_attrs[f"LABL_PTR_{depend_num}"] == f"{depend}_label"
                )
        assert ena_intensity_attrs["UNITS"] == "counts/(s * cm^2 * Sr * KeV)"

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
