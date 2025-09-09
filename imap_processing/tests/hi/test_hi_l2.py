"""Test coverage for imap_processing.hi.l2.hi_l2.py"""

from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.cdf.utils import write_cdf
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.hi.hi_l2 import (
    calculate_ena_intensity,
    calculate_ena_signal_rates,
    esa_energy_df,
    generate_hi_map,
    hi_l2,
)
from imap_processing.spice.geometry import SpiceFrame


@pytest.fixture
def empty_rectangular_map_dataset() -> xr.Dataset:
    """Generate an empty rectangular map Dataset with coords only"""
    coords = {
        "epoch": 1,
        "esa_energy_step": 3,
        "calibration_prod": 2,
        "longitude": 9,
        "latitude": 4,
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


@pytest.mark.external_test_data
@pytest.mark.external_kernel
def test_hi_l2(
    hi_l1_test_data_path,
    esa_energies_lut_path,
    geometric_factors_path,
    imap_ena_sim_metakernel,
):
    """Integration type test for hi_l2()"""
    pset_path = hi_l1_test_data_path / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"

    l2_dataset = hi_l2(
        [pset_path],
        geometric_factors_path,
        esa_energies_lut_path,
        "h90-ena-h-sf-nsp-full-hae-4deg-3mo",
    )[0]
    assert isinstance(l2_dataset, xr.Dataset)
    assert len(l2_dataset.data_vars) == 15
    np.testing.assert_array_equal(
        l2_dataset["ena_intensity"].dims, ["epoch", "energy", "longitude", "latitude"]
    )
    # Test ISTP compliance by writing the CDF
    write_cdf(l2_dataset, istp=True)


@pytest.mark.external_test_data
@patch("imap_processing.hi.hi_l2.generate_hi_map")
def test_hi_l2_uses_descriptor_to_setup_map(
    mock_generate_hi_map,
    hi_l1_test_data_path,
):
    pset_path = hi_l1_test_data_path / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"
    descriptor_str = "h90-ena-h-sf-nsp-full-hnu-2deg-3mo"
    rect_map = Mock(spec=RectangularSkyMap)
    mock_generate_hi_map.return_value = rect_map

    _ = hi_l2([pset_path], None, None, descriptor_str)[0]

    output_map = mock_generate_hi_map.call_args.kwargs["output_map"]

    assert output_map.spice_reference_frame == SpiceFrame.IMAP_HNU
    assert output_map.spacing_deg == 2.0
    assert mock_generate_hi_map.call_args.kwargs["spin_phase"] == "full"
    assert not mock_generate_hi_map.call_args.kwargs["cg_corrected"]

    rect_map.build_cdf_dataset.assert_called_with(
        "hi", "l2", "sf", descriptor_str, sensor="90"
    )


@mock.patch("imap_processing.hi.hi_l2.calculate_ena_intensity", autospec=True)
@mock.patch("imap_processing.hi.hi_l2.esa_energy_df", autospec=True)
@pytest.mark.external_test_data
def test_genarate_hi_map(
    mock_esa_energy_lookup,
    mock_calc_ena_intensity,
    hi_l1_test_data_path,
    furnish_kernels,
):
    """Test coverage for genarate_hi_map()"""

    mock_esa_energy_lookup.side_effect = lambda x, y: pd.DataFrame(
        {"nominal_central_energy": y, "bandpass_fwhm": np.ones_like(y)}
    )

    kernels = [
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "naif0012.tls",
        "imap_spk_demo.bsp",
    ]
    with furnish_kernels(kernels):
        pset_path = hi_l1_test_data_path / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"

        rectangular_sky_map = RectangularSkyMap(
            spacing_deg=6, spice_frame=SpiceFrame.IMAP_GCS
        )
        sky_map = generate_hi_map(
            [pset_path],
            None,
            None,
            rectangular_sky_map,
            cg_corrected=False,
            spin_phase="full",
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


def test_calculate_ena_intensity(
    empty_rectangular_map_dataset, esa_energies_lut_path, geometric_factors_path
):
    """Test coverage for calculate_ena_intensity"""
    # Start with an empty (coords only) dataset
    map_ds = empty_rectangular_map_dataset
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
    ena_intesity_vars = calculate_ena_intensity(
        map_ds, geometric_factors_path, esa_energies_lut_path
    )

    # TODO: add value/functional test checks once the full algorithm is implemented
    for var_name in [
        "ena_intensity",
        "ena_intensity_stat_unc",
        "ena_intensity_sys_err",
    ]:
        assert var_name in ena_intesity_vars


def test_esa_energy_lookup(esa_energies_lut_path):
    """Test coverage for esa_energy_lookup()"""
    esa_energy_steps = np.array([1, 2, 3, 3, 7, 8, 9])
    expected_energies = np.array([0.5, 0.75, 1.1, 1.1, 5.7, 8.52, 12.8])
    energy_df = esa_energy_df(esa_energies_lut_path, esa_energy_steps)
    retrieved_energies = energy_df["nominal_central_energy"].values
    np.testing.assert_array_equal(retrieved_energies, expected_energies)
    assert "bandpass_fwhm" in energy_df
