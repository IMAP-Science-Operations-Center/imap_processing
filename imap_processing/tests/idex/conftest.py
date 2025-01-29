from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.idex.idex_l1a import PacketParser

SPICE_ARRAYS = [
    "ephemeris_position_x",
    "ephemeris_position_y",
    "ephemeris_position_z",
    "ephemeris_velocity_x",
    "ephemeris_velocity_y",
    "ephemeris_velocity_z",
    "right_ascension",
    "declination",
    "solar_longitude",
    "spin_phase",
]


@pytest.fixture(scope="module")
def decom_test_data() -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xarray.Dataset
        A ``xarray`` dataset containing the test data
    """
    test_file = Path(
        f"{imap_module_directory}/tests/idex/imap_idex_l0_raw_20231214_v001.pkts"
    )
    return PacketParser(test_file, "001").data


def get_spice_data_side_effect_func(l1a_ds, idex_attrs):
    # Create a mock dictionary of spice arrays

    return {
        name: xr.DataArray(
            name=name,
            data=np.ones(len(l1a_ds["epoch"])),
            dims="epoch",
            attrs=idex_attrs.get_variable_attributes(name),
        )
        for name in SPICE_ARRAYS
    }
