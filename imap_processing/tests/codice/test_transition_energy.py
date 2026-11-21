"""Verify transition-day energy support retains its time dependence."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.codice import codice_l2, constants
from imap_processing.codice.codice_l1b import convert_to_rates


@pytest.mark.parametrize("varying", [False, True])
def test_l1b_energy_matches_each_epoch(varying):
    """L1B uses each epoch's voltage table, keeping normal days one-dimensional."""
    data = {
        name: (("epoch", "esa_step"), np.ones((2, 2)))
        for name in constants.LO_COUNTERS_SINGLES_VARIABLE_NAMES
    }
    for name in [
        "nso_half_spin",
        "sw_bias_gain_mode",
        "st_bias_gain_mode",
        "spin_period",
        "nso_esa_step",
        "nso_spin_sector",
    ]:
        data[name] = ("epoch", [1, 1])
    data["acquisition_time_per_esa_step"] = (("epoch", "esa_step"), np.ones((2, 2)))
    dataset = xr.Dataset(data, coords={"epoch": [10, 20], "esa_step": [0, 1]})
    dataset["k_factor"] = xr.DataArray([2.0], dims="k_factor")
    voltage = [[100.0, 200.0], [300.0, 400.0]] if varying else [100.0, 200.0]
    dims = ("epoch", "esa_step") if varying else ("esa_step",)
    dataset["voltage_table"] = xr.DataArray(voltage, dims=dims)
    result = convert_to_rates(dataset, "lo-counters-singles")
    assert result["energy_per_charge"].dims == dims
    np.testing.assert_allclose(result["energy_per_charge"], np.asarray(voltage) * 0.002)
    assert result["energy_per_charge_label"].dims == dims
    if varying:
        assert result["energy_per_charge"].attrs["DEPEND_0"] == "epoch"
        assert result["energy_per_charge_label"].values.tolist() == [
            ["0.200", "0.400"],
            ["0.600", "0.800"],
        ]


@pytest.mark.parametrize("varying", [False, True])
def test_l2_preserves_time_dependent_energy(monkeypatch, varying):
    """The L2 entry point must not swap an axis to a two-dimensional coordinate."""

    dims = ("epoch", "esa_step") if varying else ("esa_step",)
    values = np.array([[0.2, 0.4], [0.6, 0.8]]) if varying else np.array([0.2, 0.4])
    dataset = xr.Dataset(
        {
            "energy_per_charge": (dims, values, {"DEPEND_1": "esa_step"}),
            "energy_per_charge_label": (dims, np.char.mod("%.3f", values)),
            "esa_step_label": ("esa_step", ["0", "1"]),
            "counts": (("epoch", "esa_step"), np.ones((2, 2))),
        },
        coords={"epoch": [10, 20], "esa_step": [0, 1]},
    )
    if varying:
        dataset["energy_per_charge"].attrs["DEPEND_0"] = "epoch"
    dependencies = Mock()
    dependencies.get_file_paths.return_value = [
        "imap_codice_l1b_lo-sw-species_20251218_v001.0001.cdf"
    ]
    monkeypatch.setattr(codice_l2, "load_cdf", lambda _: dataset)
    monkeypatch.setattr(codice_l2, "get_geometric_factor_lut", lambda _: None)
    monkeypatch.setattr(codice_l2, "compute_geometric_factors", lambda *_: None)
    monkeypatch.setattr(
        codice_l2, "get_efficiency_lut", lambda _: pd.DataFrame({"product": ["sw"]})
    )
    # Isolate coordinate handling; intensity math already uses named dimensions.
    monkeypatch.setattr(codice_l2, "process_lo_species_intensity", lambda ds, *_: ds)
    result = codice_l2.process_codice_l2("lo-sw-species", dependencies)
    np.testing.assert_array_equal(result["energy_per_charge"], values)
    if varying:
        assert result["energy_per_charge"].dims == ("epoch", "esa_step")
        assert result["counts"].dims == ("epoch", "esa_step")
        assert result["energy_per_charge"].attrs["DEPEND_0"] == "epoch"
    else:
        assert result["counts"].dims == ("epoch", "energy_per_charge")
