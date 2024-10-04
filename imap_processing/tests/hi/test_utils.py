"""Test coverage for imap_processing.hi.utils.py"""

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hi.utils import HIAPID, create_dataset_variables, full_dataarray


def test_hiapid():
    """Test coverage for HIAPID class"""
    hi_apid = HIAPID(754)
    assert isinstance(hi_apid, HIAPID)
    assert hi_apid.name == "H45_APP_NHK"
    assert hi_apid.sensor == "45sensor"

    hi_apid = HIAPID["H90_SCI_CNT"]
    assert hi_apid.value == 833
    assert hi_apid.sensor == "90sensor"


@pytest.mark.parametrize(
    "name, shape, expected_shape",
    [
        ("despun_z", (1, 3), (1, 3)),
        ("hae_latitude", None, (1, 360)),
        ("counts", None, (1, 10, 360)),
    ],
)
def test_full_dataarray(name, shape, expected_shape):
    """Test coverage for full_dataarray function"""
    coords = {
        "epoch": xr.DataArray(np.array([0])),
        "esa_energy_step": xr.DataArray(np.arange(10)),
        "spin_angle_bin": xr.DataArray(np.arange(360)),
    }
    cdf_manager = ImapCdfAttributes()
    cdf_manager.load_variable_attributes("imap_hi_variable_attrs.yaml")

    dataarray = full_dataarray(
        name, cdf_manager.get_variable_attributes(f"hi_pset_{name}"), coords, shape
    )
    assert dataarray.data.shape == expected_shape


@pytest.mark.parametrize(
    "var_names, shape, lookup_str",
    [
        (["delta_t_ab", "delta_t_ac1"], 5, "hi_de_{0}"),
        (["hae_latitude"], (3, 5), "hi_pset_{0}"),
    ],
)
def test_create_dataset_variables(var_names, shape, lookup_str):
    """Test coverage for `imap_processing.hi.utils.create_l1b_de_variables`"""
    shape = 5
    var_names = ["delta_t_ab", "delta_t_ac1", "delta_t_bc1"]
    l1b_de_vars = create_dataset_variables(
        var_names, shape, att_manager_lookup_str="hi_de_{0}"
    )
    assert len(l1b_de_vars) == len(var_names)
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.load_variable_attributes("imap_hi_variable_attrs.yaml")

    for var_name, data_array in l1b_de_vars.items():
        attrs = attr_mgr.get_variable_attributes(
            f"hi_de_{var_name}", check_schema=False
        )
        assert data_array.values.dtype == attrs["dtype"]
        if data_array.ndim == 1:
            assert data_array.size == shape
        else:
            assert data_array.shape == shape
