"""Test coverage for imap_processing.hi.l2.hi_l2.py"""

import numpy as np
import pytest
import xarray as xr

from imap_processing.hi.l2.hi_l2 import (
    calculate_ena_intensity,
    calculate_ena_signal_rates,
    generate_hi_map,
)


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
                np.arange(v),
                name=k,
                dims=[k],
            )
            for k, v in coords.items()
        },
    )
    return map_ds


@pytest.mark.external_test_data
def test_hi_l2(hi_l1_test_data_path):
    """Integration type test for hi_l2()"""
    pset_path = hi_l1_test_data_path / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"
    l2_dataset = generate_hi_map([pset_path], None, None)
    assert isinstance(l2_dataset, xr.Dataset)


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


def test_calculate_ena_intensity(empty_rectangular_map_dataset):
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
    ena_intesity_vars = calculate_ena_intensity(map_ds, None, None)

    # TODO: add value/functional test checks once the full algorithm is implemented
    for var_name in [
        "ena_intensity",
        "ena_intensity_stat_unc",
        "ena_intensity_sys_err",
    ]:
        assert var_name in ena_intesity_vars
