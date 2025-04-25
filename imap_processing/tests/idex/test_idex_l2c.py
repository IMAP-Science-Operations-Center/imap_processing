"""Tests the L2c processing for IDEX data"""

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.idex.idex_l2c import idex_l2c


@pytest.fixture
@mock.patch("imap_processing.idex.idex_l1b.get_spice_data")
def l2c_dataset(decom_test_data: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """

    longitude = xr.DataArray(
        np.random.uniform(0, 360, len(decom_test_data.epoch)),
        dims=["epoch"],
        name="longitude",
    )
    latitude = xr.DataArray(
        np.random.uniform(-90, 90, len(decom_test_data.epoch)),
        dims=["epoch"],
        name="latitude",
    )
    with mock.patch(
        "imap_processing.idex.idex_l1b.get_spice_data",
        return_value={"longitude": longitude, "latitude": latitude},
    ):
        dataset = idex_l2c(idex_l1b(decom_test_data))
    return dataset


def test_l2c_logical_source(l2c_dataset: xr.Dataset):
    """Tests that the ``idex_l2b`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    l2c_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_src = "imap_idex_l2c_sci"
    assert l2c_dataset.attrs["Logical_source"] == expected_src
