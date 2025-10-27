"""Test coverage for imap_processing.hi.l2.hi_l2.py"""

from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import write_cdf
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.hi.hi_l2 import (
    _calculate_improved_stat_variance,
    calculate_ena_intensity,
    calculate_ena_signal_rates,
    combine_calibration_products,
    esa_energy_df,
    generate_hi_map,
    hi_l2,
)
from imap_processing.spice.geometry import SpiceFrame


@pytest.fixture(scope="module")
def empty_rectangular_map_dataset() -> xr.Dataset:
    """Generate an empty rectangular map Dataset with coords only"""
    coords = {
        "epoch": 1,
        "esa_energy_step": 9,
        "calibration_prod": 2,
        "spatial": 12,
    }
    map_ds = xr.Dataset(
        coords={
            k: xr.DataArray(
                np.arange(v) + 1 if k == "esa_energy_step" else np.arange(v),
                name=k,
                dims=[k],
            )
            for k, v in coords.items()
        },
    )
    return map_ds


@pytest.fixture
def esa_energies_lut_path(hi_l1_test_data_path):
    return hi_l1_test_data_path / "imap_hi_90sensor-esa-energies_20240101_v001.csv"


@pytest.fixture
def geometric_factors_path(hi_l1_test_data_path):
    return hi_l1_test_data_path / "imap_hi_90sensor-cal-prod_20240101_v001.csv"


@pytest.fixture
def esa_eta_fit_factors_path(imap_tests_path):
    return (
        imap_tests_path
        / "ena_maps/data/imap_hi_90sensor-esa-eta-fit-factors_20240101_v001.csv"
    )


@pytest.fixture
def anc_path_dict(
    esa_energies_lut_path, geometric_factors_path, esa_eta_fit_factors_path
):
    path_dict = {
        "cal-prod": geometric_factors_path,
        "esa-energies": esa_energies_lut_path,
        "esa-eta-fit-factors": esa_eta_fit_factors_path,
    }
    return path_dict


@pytest.fixture
def sample_map_dataset():
    """Generate a realistic map dataset for testing calibration product combining"""
    coords = {
        "epoch": 1,
        "esa_energy_step": 3,
        "calibration_prod": 3,
        "longitude": 4,
        "latitude": 2,
    }

    coords = {
        k: xr.DataArray(
            np.arange(v) + 1 if k == "esa_energy_step" else np.arange(v),
            name=k,
            dims=[k],
        )
        for k, v in coords.items()
    }

    # Create fake geometric factors (different sensitivities)
    geometric_factors = xr.DataArray(
        np.array([[1.0, 2.0, 0.5], [1.5, 3.0, 0.75], [2.0, 4.0, 1.0]]),
        dims=["esa_energy_step", "calibration_prod"],
    )

    # ESA energies
    esa_energies = xr.DataArray(
        np.array([0.5, 0.75, 1.1]),
        dims=["esa_energy_step"],
    )

    # Create sample data with some realistic structure
    shape = tuple(darray.size for darray in coords.values())

    np.random.seed(42)  # For reproducible tests

    # Create test dataset
    test_ds = xr.Dataset(
        {
            "ena_signal_rates": xr.DataArray(
                np.random.rand(*shape) * 1000 + 100, dims=list(coords.keys())
            ),
            "ena_intensity": xr.DataArray(
                np.random.rand(*shape) * 100 + 50, dims=list(coords.keys())
            ),
            "ena_intensity_stat_uncert": xr.DataArray(
                np.random.rand(*shape) * 10 + 5, dims=list(coords.keys())
            ),
            "ena_intensity_sys_err": xr.DataArray(
                np.random.rand(*shape) * 5 + 1, dims=list(coords.keys())
            ),
            "bg_rates": xr.DataArray(
                np.random.rand(*shape) * 20 + 5, dims=list(coords.keys())
            ),
            "bg_rates_unc": xr.DataArray(
                np.random.rand(*shape) * 2 + 1, dims=list(coords.keys())
            ),
            "exposure_factor": xr.DataArray(
                np.random.rand(*shape) * 5 + 1, dims=list(coords.keys())
            ),
        },
        coords=coords,
    )

    return test_ds, geometric_factors, esa_energies


@pytest.mark.parametrize(
    "descriptor_str",
    [
        "h90-ena-h-sf-nsp-full-hae-4deg-3mo",
        "h90-ena-h-hf-nsp-ram-gcs-6deg-3mo",
    ],
)
@pytest.mark.external_test_data
@pytest.mark.external_kernel
def test_hi_l2(
    descriptor_str,
    hi_l1_test_data_path,
    anc_path_dict,
    imap_ena_sim_metakernel,
):
    """Integration type test for hi_l2()"""
    pset_path = hi_l1_test_data_path / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"

    l2_dataset = hi_l2(
        [pset_path],
        anc_path_dict,
        descriptor_str,
    )[0]
    assert isinstance(l2_dataset, xr.Dataset)

    # Check some global attributes
    assert l2_dataset.attrs["Data_type"].startswith(f"L2_{descriptor_str}")
    assert l2_dataset.attrs["Logical_source"] == f"imap_hi_l2_{descriptor_str}"
    assert "Hi90" in l2_dataset.attrs["Logical_source_description"]

    assert len(l2_dataset.data_vars) == 15
    np.testing.assert_array_equal(
        l2_dataset["ena_intensity"].dims, ["epoch", "energy", "longitude", "latitude"]
    )
    # Test ISTP compliance by writing the CDF
    write_cdf(l2_dataset, istp=True)


@pytest.mark.external_test_data
@patch(
    "imap_processing.ena_maps.ena_maps.RectangularSkyMap.build_cdf_dataset",
    autospec=True,
)
@patch("imap_processing.hi.hi_l2.generate_hi_map")
def test_hi_l2_uses_descriptor_to_setup_map(
    mock_generate_hi_map,
    mock_map_build_cdf_dataset,
    hi_l1_test_data_path,
):
    pset_path = hi_l1_test_data_path / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"
    descriptor_str = "h90-ena-h-sf-nsp-full-hnu-2deg-3mo"
    rect_map = MapDescriptor.from_string(descriptor_str).to_empty_map()
    mock_generate_hi_map.return_value = rect_map
    mock_map_build_cdf_dataset.return_value = xr.Dataset()

    _ = hi_l2([pset_path], None, descriptor_str)[0]

    assert rect_map.spice_reference_frame == SpiceFrame.IMAP_HNU
    assert rect_map.spacing_deg == 2.0

    mock_map_build_cdf_dataset.assert_called_with(
        rect_map, "hi", "l2", descriptor_str, sensor="90"
    )


@pytest.mark.parametrize(
    "descriptor_str",
    [
        "h90-ena-h-sf-nsp-full-gcs-6deg-3mo",
        "h90-ena-h-sf-nsp-ram-gcs-6deg-3mo",
        "h90-ena-h-hf-nsp-ram-gcs-6deg-3mo",
    ],
)
@mock.patch("imap_processing.hi.hi_l2.calculate_ena_intensity", autospec=True)
@mock.patch(
    "imap_processing.hi.hi_l2.interpolate_map_flux_to_helio_frame", autospec=True
)
@pytest.mark.external_test_data
def test_genarate_hi_map(
    mock_interp_flux,
    mock_calc_ena_intensity,
    hi_l1_test_data_path,
    anc_path_dict,
    furnish_kernels,
    descriptor_str,
):
    """Test coverage for genarate_hi_map()"""
    mock_calc_ena_intensity.side_effect = lambda x, y, z: x
    mock_interp_flux.side_effect = lambda x, y, z: x

    kernels = [
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "naif0012.tls",
        "imap_spk_demo.bsp",
        "de440s.bsp",
    ]
    with furnish_kernels(kernels):
        pset_path = hi_l1_test_data_path / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"

        rect_map = MapDescriptor.from_string(descriptor_str)
        sky_map = generate_hi_map(
            [pset_path],
            anc_path_dict,
            rect_map,
        )
    assert isinstance(sky_map, RectangularSkyMap)
    assert sky_map.spacing_deg == 6
    assert sky_map.spice_reference_frame == SpiceFrame.IMAP_GCS

    # Check that calculate_ena_intensities was called
    mock_calc_ena_intensity.assert_called_once()

    # Test that we got some non-zero values
    for var_name in ["counts", "exposure_factor", "obs_date"]:
        assert var_name in sky_map.data_1d.data_vars
        assert np.nanmax(sky_map.data_1d[var_name].data) > 0
    # If the CG correction ran, check that the energy_sc variable is present
    # in the map
    if "-hf-" in descriptor_str:
        assert "energy_sc" in sky_map.data_1d.data_vars
        assert np.nanmax(sky_map.data_1d["energy_sc"].data) > 0
        mock_interp_flux.assert_called_once()


def test_calculate_ena_signal_rates(empty_rectangular_map_dataset):
    """Test coverage for calculate_ena_signal_rates"""
    # Start with an empty (coords only) dataset
    map_ds = empty_rectangular_map_dataset
    # Add some data_vars needed for the signal rates calculations
    counts_shape = tuple(map_ds.sizes.values())
    exposure_sizes = {k: v for k, v in map_ds.sizes.items() if k != "calibration_prod"}
    # By using np.arange % n_i where no n shares a common factor with any other n,
    # we ensure that each unique combination is encountered in a PSET bin.
    map_ds.update(
        {
            "counts": xr.DataArray(
                np.arange(np.prod(tuple(map_ds.sizes.values()))).reshape(counts_shape)
                % 5,
                name="counts",
                dims=list(map_ds.sizes.keys()),
            ),
            "exposure_factor": xr.DataArray(
                np.arange(np.prod(tuple(exposure_sizes.values()))).reshape(
                    tuple(exposure_sizes.values())
                )
                % 3,
                name="exposure_factor",
                dims=list(exposure_sizes.keys()),
            ),
            "bg_rates": xr.DataArray(
                np.arange(np.prod(tuple(map_ds.sizes.values()))).reshape(counts_shape)
                % 2,
                name="bg_rates",
                dims=list(map_ds.sizes.keys()),
            ),
        }
    )
    signal_rates_vars = calculate_ena_signal_rates(map_ds)
    for var_name in ["ena_signal_rates", "ena_signal_rate_stat_unc"]:
        assert var_name in signal_rates_vars
        assert signal_rates_vars[var_name].shape == counts_shape
    # Verify that there are no negative signal rates. The synthetic data combination
    # where counts = 0, exposure_factor = 1, and bg_rates = 1 would result in
    # an ena_signal_rate of (0 / 1) - 1 = -1
    assert np.nanmin(signal_rates_vars["ena_signal_rates"].values) >= 0
    # Verify that the minimum finite uncertainty is sqrt(1) / exposure_factor.
    # The max exposure factor is 2, so we can expect the minimum finite
    # uncertainty value to be 1/2.
    assert np.nanmin(signal_rates_vars["ena_signal_rate_stat_unc"].values) == 1 / 2


@pytest.fixture(scope="module")
def ena_intensity_map_ds(empty_rectangular_map_dataset):
    """Fixture that produces a dataset to use in testing ena_intensity."""
    # Start with an empty (coords only) dataset
    map_ds = empty_rectangular_map_dataset.copy()
    # Add some data_vars needed for the ena intensity calculations
    var_shape = tuple(map_ds.sizes.values())
    map_ds.update(
        {
            "ena_signal_rates": xr.DataArray(
                np.arange(np.prod(tuple(map_ds.sizes.values()))).reshape(var_shape) % 5,
                name="ena_signal_rates",
                dims=list(map_ds.sizes.keys()),
            ),
            "ena_signal_rate_stat_unc": xr.DataArray(
                np.arange(np.prod(tuple(map_ds.sizes.values()))).reshape(var_shape) % 4
                + 1,
                name="ena_signal_rate_stat_unc",
                dims=list(map_ds.sizes.keys()),
            ),
            "bg_rates_unc": xr.DataArray(
                np.arange(np.prod(tuple(map_ds.sizes.values()))).reshape(var_shape) % 3,
                name="bg_rates_unc",
                dims=list(map_ds.sizes.keys()),
            ),
        }
    )

    # Add required background data
    bg_shape = tuple(
        map_ds.sizes[k] for k in map_ds.sizes.keys() if k != "calibration_prod"
    )
    map_ds.update(
        {
            "bg_rates": xr.DataArray(
                np.ones(bg_shape) * 5.0,
                dims=[d for d in map_ds.sizes.keys() if d != "calibration_prod"],
            ),
            "exposure_factor": xr.DataArray(
                np.ones(bg_shape) * 2.0,
                dims=[d for d in map_ds.sizes.keys() if d != "calibration_prod"],
            ),
        }
    )
    return map_ds


def test_calculate_ena_intensity(ena_intensity_map_ds, anc_path_dict):
    """Test coverage for calculate_ena_intensity"""
    descriptor_str = "h90-ena-h-sf-nsp-full-gcs-6deg-3mo"
    map_descriptor = MapDescriptor.from_string(descriptor_str)

    result_ds = calculate_ena_intensity(
        ena_intensity_map_ds, anc_path_dict, map_descriptor
    )

    for var_name in [
        "ena_intensity",
        "ena_intensity_stat_uncert",
        "ena_intensity_sys_err",
    ]:
        assert var_name in result_ds
        # Check that calibration_prod dimension has been removed
        assert "calibration_prod" not in result_ds[var_name].dims


@pytest.mark.parametrize(
    "descriptor_str, flux_corrected",
    [
        ("h90-ena-h-sf-nsp-anti-gcs-6deg-3mo", True),
        ("h90-enaraw-h-hf-nsp-ram-gcs-6deg-3mo", False),
    ],
)
@mock.patch("imap_processing.hi.hi_l2.PowerLawFluxCorrector", autospec=True)
def test_calculate_ena_intensity_flux_correction_logic(
    mock_flux_corrector_class,
    descriptor_str,
    flux_corrected,
    ena_intensity_map_ds,
    anc_path_dict,
):
    """Test that flux correction is applied based on map descriptor."""
    # Create a mock instance that will be returned when PowerLawFluxCorrector
    # is instantiated
    mock_instance = mock_flux_corrector_class.return_value
    mock_instance.apply_flux_correction.side_effect = (
        lambda intensity, stat_unc, energy: (intensity, stat_unc)
    )

    map_descriptor = MapDescriptor.from_string(descriptor_str)
    _ = calculate_ena_intensity(ena_intensity_map_ds, anc_path_dict, map_descriptor)

    # Now check if the method was called based on the flux_corrected expectation
    if flux_corrected:
        mock_instance.apply_flux_correction.assert_called_once()
    else:
        mock_instance.apply_flux_correction.assert_not_called()


def test_combine_calibration_products(sample_map_dataset):
    """Test coverage for combine_calibration_products"""
    test_ds, geometric_factors, esa_energies = sample_map_dataset

    # Make a copy to avoid modifying the fixture
    test_ds_copy = test_ds.copy(deep=True)

    result_ds = combine_calibration_products(
        test_ds_copy, geometric_factors, esa_energies
    )

    # Check that all expected variables are present
    expected_vars = [
        "ena_intensity",
        "ena_intensity_stat_uncert",
        "ena_intensity_sys_err",
    ]
    for var_name in expected_vars:
        assert var_name in result_ds
        # Check that calibration_prod dimension has been removed
        assert "calibration_prod" not in result_ds[var_name].dims
        # Check that other dimensions are preserved
        expected_dims = [
            d for d in test_ds["ena_intensity"].dims if d != "calibration_prod"
        ]
        assert list(result_ds[var_name].dims) == expected_dims

    # Check that combined flux is finite where input data is valid
    combined_flux = result_ds["ena_intensity"]
    assert np.any(np.isfinite(combined_flux.values)), (
        "No valid combined flux values produced"
    )

    # Check that combined uncertainty is reasonable
    combined_unc = result_ds["ena_intensity_stat_uncert"]
    assert np.all(combined_unc.values[np.isfinite(combined_unc.values)] >= 0), (
        "Combined uncertainty should be non-negative"
    )

    # Check systematic error combination (root sum of squares)
    input_sys_err = test_ds["ena_intensity_sys_err"]
    expected_sys_err = np.sqrt((input_sys_err**2).sum(dim="calibration_prod"))
    combined_sys_err = result_ds["ena_intensity_sys_err"]

    np.testing.assert_array_almost_equal(
        combined_sys_err.values, expected_sys_err.values, decimal=10
    )


def test_calculate_improved_variance(sample_map_dataset):
    """Test _calculate_improved_stat_variance function"""
    test_ds, geometric_factors, esa_energies = sample_map_dataset

    improved_unc = _calculate_improved_stat_variance(
        test_ds, geometric_factors, esa_energies
    )

    # Check that result has same shape as input statistical uncertainties
    original_unc = test_ds["ena_intensity_stat_uncert"]
    assert improved_unc.shape == original_unc.shape

    # Check that improved uncertainties are finite and non-negative
    assert np.all(improved_unc.values >= 0)
    assert np.all(np.isfinite(improved_unc.values))


def test_calculate_improved_variance_single_product():
    """Test improved variance with single calibration product"""
    coords = {
        "epoch": 1,
        "esa_energy_step": 1,
        "calibration_prod": 1,
        "longitude": 1,
        "latitude": 1,
    }

    geom_factors = xr.DataArray(
        np.array([[1.0]]), dims=["esa_energy_step", "calibration_prod"]
    )

    esa_energies = xr.DataArray(np.array([1.0]), dims=["esa_energy_step"])

    test_ds = xr.Dataset(
        {
            "ena_intensity": xr.DataArray(
                np.array([[[[[100.0]]]]]), dims=list(coords.keys())
            ),
            "ena_intensity_stat_uncert": xr.DataArray(
                np.array([[[[[10.0]]]]]), dims=list(coords.keys())
            ),
        }
    )

    improved_var = _calculate_improved_stat_variance(
        test_ds, geom_factors, esa_energies
    )

    # With single product, should return original uncertainties
    np.testing.assert_array_equal(
        improved_var.values, test_ds["ena_intensity_stat_uncert"].values ** 2
    )


def test_esa_energy_lookup(esa_energies_lut_path):
    """Test coverage for esa_energy_df()"""
    esa_energy_steps = np.array([1, 2, 3, 3, 7, 8, 9])
    expected_energies = np.array([0.5, 0.75, 1.1, 1.1, 5.7, 8.52, 12.8])
    energy_df = esa_energy_df(esa_energies_lut_path, esa_energy_steps)
    retrieved_energies = energy_df["nominal_central_energy"].values
    np.testing.assert_array_equal(retrieved_energies, expected_energies)
    assert "bandpass_fwhm" in energy_df


def test_weighted_average_mathematical_correctness():
    """Test mathematical properties of the weighted average"""
    # Create a simple test case where we can verify the weighted average manually
    coords = {
        "epoch": 1,
        "esa_energy_step": 1,
        "calibration_prod": 2,
        "longitude": 1,
        "latitude": 1,
    }

    # Simple test data
    flux_values = np.array([100.0, 200.0]).reshape(1, 1, 2, 1, 1)
    stat_unc_values = np.array([10.0, 20.0]).reshape(1, 1, 2, 1, 1)
    sys_err_values = np.array([5.0, 10.0]).reshape(1, 1, 2, 1, 1)

    test_ds = xr.Dataset(
        {
            "ena_intensity": xr.DataArray(flux_values, dims=list(coords.keys())),
            "ena_intensity_stat_uncert": xr.DataArray(
                stat_unc_values, dims=list(coords.keys())
            ),
            "ena_intensity_sys_err": xr.DataArray(
                sys_err_values, dims=list(coords.keys())
            ),
            "ena_signal_rates": xr.DataArray(
                np.array([100.0, 400.0]).reshape(1, 1, 2, 1, 1),
                dims=list(coords.keys()),
            ),
            "bg_rates": xr.DataArray(
                np.array([5.0]).reshape(1, 1, 1, 1),
                dims=[d for d in coords.keys() if d != "calibration_prod"],
            ),
            "exposure_factor": xr.DataArray(
                np.array([2.0]).reshape(1, 1, 1, 1),
                dims=[d for d in coords.keys() if d != "calibration_prod"],
            ),
        }
    )

    geom_factors = xr.DataArray(
        np.array([[1.0, 2.0]]), dims=["esa_energy_step", "calibration_prod"]
    )

    esa_energies = xr.DataArray(np.array([1.0]), dims=["esa_energy_step"])

    result_ds = combine_calibration_products(test_ds, geom_factors, esa_energies)

    # Check that results are finite and reasonable
    assert np.isfinite(result_ds["ena_intensity"].values[0, 0, 0, 0])
    assert result_ds["ena_intensity_stat_uncert"].values[0, 0, 0, 0] > 0
    assert result_ds["ena_intensity_sys_err"].values[0, 0, 0, 0] > 0

    # Systematic error should be root sum of squares
    expected_sys_err = np.sqrt(5.0**2 + 10.0**2)
    np.testing.assert_almost_equal(
        result_ds["ena_intensity_sys_err"].values[0, 0, 0, 0],
        expected_sys_err,
        decimal=10,
    )


def test_statistical_uncertainty_combination_correctness():
    """Test that statistical uncertainties are combined correctly."""
    coords = {
        "epoch": 1,
        "esa_energy_step": 1,
        "calibration_prod": 2,
        "longitude": 1,
        "latitude": 1,
    }

    # Create simple case with known statistical uncertainties
    stat_unc_values = np.array([5.0, 10.0]).reshape(1, 1, 2, 1, 1)
    sys_err_values = np.array([2.0, 4.0]).reshape(1, 1, 2, 1, 1)
    flux_values = np.array([90.0, 210.0]).reshape(1, 1, 2, 1, 1)

    test_ds = xr.Dataset(
        {
            "ena_intensity": xr.DataArray(flux_values, dims=list(coords.keys())),
            "ena_intensity_stat_uncert": xr.DataArray(
                stat_unc_values, dims=list(coords.keys())
            ),
            "ena_intensity_sys_err": xr.DataArray(
                sys_err_values, dims=list(coords.keys())
            ),
            "ena_signal_rates": xr.DataArray(flux_values, dims=list(coords.keys())),
            "bg_rates": xr.DataArray(
                np.array([1.0, 2.0]).reshape(1, 1, 2, 1, 1), dims=list(coords.keys())
            ),
            "exposure_factor": xr.DataArray(
                np.array([1.0, 1.0]).reshape(1, 1, 2, 1, 1), dims=list(coords.keys())
            ),
        }
    )

    geom_factors = xr.DataArray(
        np.array([[1.0, 2.0]]),
        dims=["esa_energy_step", "calibration_prod"],
    )

    esa_energies = xr.DataArray(np.array([1.0]), dims=["esa_energy_step"])

    result_ds = combine_calibration_products(test_ds, geom_factors, esa_energies)

    # Manual calculation of expected statistical uncertainty combination
    # combined_stat_unc = sqrt(1/sum(1 / stat_unc**2))
    expected_combined_stat_unc = np.sqrt(1 / np.sum(1 / stat_unc_values**2))
    flux_weights = 1.0 / (np.array([101, 101]) + np.array([4, 16]))
    expected_flux = np.sum(flux_values.squeeze() * flux_weights) / np.sum(flux_weights)

    np.testing.assert_almost_equal(
        result_ds["ena_intensity_stat_uncert"].values,
        expected_combined_stat_unc,
        decimal=10,
    )

    np.testing.assert_almost_equal(
        result_ds["ena_intensity"].values, expected_flux, decimal=10
    )


def test_combine_calibration_products_edge_cases():
    """Test edge cases for combine_calibration_products"""
    # Test with single calibration product
    coords = {
        "epoch": 1,
        "esa_energy_step": 1,
        "calibration_prod": 1,
        "longitude": 1,
        "latitude": 1,
    }

    test_ds = xr.Dataset(
        {
            "ena_intensity": xr.DataArray(
                np.array([100.0]).reshape(1, 1, 1, 1, 1), dims=list(coords.keys())
            ),
            "ena_intensity_stat_uncert": xr.DataArray(
                np.array([10.0]).reshape(1, 1, 1, 1, 1), dims=list(coords.keys())
            ),
            "ena_intensity_sys_err": xr.DataArray(
                np.array([5.0]).reshape(1, 1, 1, 1, 1), dims=list(coords.keys())
            ),
        }
    )

    geom_factors = xr.DataArray(
        np.array([[1.0]]), dims=["esa_energy_step", "calibration_prod"]
    )

    esa_energies = xr.DataArray(np.array([1.0]), dims=["esa_energy_step"])

    result_ds = combine_calibration_products(test_ds, geom_factors, esa_energies)

    # With single calibration product, should return dataset unchanged
    # (but without calib_prod dim). The intensity value should be the same.
    np.testing.assert_almost_equal(result_ds["ena_intensity"].values[0, 0, 0, 0], 100.0)

    # Check that calibration_prod dimension was removed
    for var in ["ena_intensity", "ena_intensity_stat_uncert", "ena_intensity_sys_err"]:
        assert "calibration_prod" not in result_ds[var].dims
