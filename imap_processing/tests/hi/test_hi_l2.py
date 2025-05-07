"""Test coverage for imap_processing.hi.l2.hi_l2.py"""

import numpy as np
import pytest
import xarray as xr

from imap_processing.hi.l2.hi_l2 import calculate_ena_signal_rates


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


def test_calculate_ena_signal_rates(empty_rectangular_map_dataset):
    """Test coverage for calculate_ena_signal_rates"""
    map_ds = empty_rectangular_map_dataset
    counts_shape = tuple(map_ds.sizes.values())
    exposure_sizes = {k: v for k, v in map_ds.sizes.items() if k != "calibration_prod"}
    map_ds.update(
        {
            "counts": xr.DataArray(
                np.arange(np.prod(tuple(map_ds.sizes.values()))).reshape(counts_shape)
                % 3,
                name="counts",
                dims=list(map_ds.sizes.keys()),
            ),
            "exposure_factor": xr.DataArray(
                np.arange(np.prod(tuple(exposure_sizes.values()))).reshape(
                    tuple(exposure_sizes.values())
                )
                % 2,
                name="exposure_factor",
                dims=list(exposure_sizes.keys()),
            ),
            "background_rates": xr.DataArray(
                np.arange(np.prod(tuple(exposure_sizes.values()))).reshape(
                    tuple(exposure_sizes.values())
                )
                % 2,
                name="background_rates",
                dims=list(exposure_sizes.keys()),
            ),
        }
    )
    signal_rates_vars = calculate_ena_signal_rates(map_ds)
    for var_name in ["ena_signal_rates", "ena_signal_rate_stat_unc"]:
        assert var_name in signal_rates_vars
