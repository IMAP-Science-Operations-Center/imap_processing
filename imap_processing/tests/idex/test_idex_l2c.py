"""Tests the L2c processing for IDEX data"""

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import write_cdf
from imap_processing.idex.idex_constants import (
    IDEX_SPACING_DEG,
)
from imap_processing.idex.idex_l2c import idex_l2c


@pytest.fixture
def l2c_dataset(l2b_dataset: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : list[xr.Dataset]
        A list of ``xarray`` datasets containing the test data
    """

    return idex_l2c([l2b_dataset])


def test_l2c_attrs_and_vars(l2c_dataset: xr.Dataset, l2a_dataset: xr.Dataset):
    """Tests that the ``idex_l2b`` function generates datasets
    with the expected variables and attributes.

    Parameters
    ----------
    l2c_dataset : xr.Dataset
        A ``xarray`` dataset containing the l2c test data.
    l2a_dataset
        A ``xarray`` dataset containing the l1b test data.
    """
    assert l2c_dataset.attrs["Logical_source"] == "imap_idex_l2c_rectangular-map-1mo"
    # The total counts in the map should be equal to the number of dust events
    # in the l2a_dataset (*2 because the l2b fixture counts are doubled)
    np.testing.assert_allclose(
        l2c_dataset["counts_by_charge_map"].sum(), len(l2a_dataset.epoch) * 2
    )
    np.testing.assert_allclose(
        l2c_dataset["counts_by_mass_map"].sum(), len(l2a_dataset.epoch) * 2
    )
    assert l2c_dataset.sizes == {
        "epoch": 2,
        "impact_charge_bins": 11,
        "mass_bins": 11,
        "rectangular_lon_pixel": int(360 / IDEX_SPACING_DEG),
        "rectangular_lat_pixel": int(180 / IDEX_SPACING_DEG),
    }
    l2c_dataset.attrs["Data_version"] = "999"
    # Check the attributes of the dataset by writing to a CDF file
    rect_file_name = write_cdf(l2c_dataset)
    assert rect_file_name.exists()
    assert rect_file_name.name == "imap_idex_l2c_rectangular-map-1mo_20251017_v999.cdf"

    for var in l2c_dataset.data_vars:
        assert "DICT_KEY" in l2c_dataset[var].attrs, (
            f"Variable {var} is missing the DICT_KEY attribute for SPASE metadata."
        )
