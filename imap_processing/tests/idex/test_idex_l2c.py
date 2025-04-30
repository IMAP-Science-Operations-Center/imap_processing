"""Tests the L2c processing for IDEX data"""

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.idex.idex_constants import IDEX_HEALPIX_NSIDE
from imap_processing.idex.idex_l2c import idex_l2c, idex_pset


@pytest.fixture
def l2c_dataset(l1b_dataset: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """

    return idex_l2c([l1b_dataset, l1b_dataset])


def test_l2c_attrs_and_vars(l2c_dataset: xr.Dataset, l1b_dataset: xr.Dataset):
    """Tests that the ``idex_l2b`` function generates datasets
    with the expected variables and attributes.

    Parameters
    ----------
    l2c_dataset : xr.Dataset
        A ``xarray`` dataset containing the l2c test data.
    l1b_dataset
        A ``xarray`` dataset containing the l1b test data.
    """
    expected_src = "imap_idex_l2c_sci"
    assert l2c_dataset.attrs["Logical_source"] == expected_src

    # The total counts in the skymap should be equal to the number of dust events
    # in the l1b_dataset (x2 because there are two identical l1b datasets in the skymap)
    np.testing.assert_allclose(l2c_dataset["counts"].sum(), len(l1b_dataset.epoch) * 2)
    assert l2c_dataset.dims == {"healpix_index": 768}

    # Check the attributes of the dataset by writing to a CDF file
    # TODO map does not have epoch and can not be written out
    # write_cdf(l2c_dataset)


def test_idex_pset(l1b_dataset: xr.Dataset):
    """Test for idex_pset function"""
    idex_attr = ImapCdfAttributes()
    idex_attr.add_instrument_global_attrs("idex")
    # idex_attr.add_instrument_variable_attrs("idex", "l2c")
    pset = idex_pset(l1b_dataset, idex_attr)

    assert pset.epoch == np.mean([l1b_dataset.epoch[0], l1b_dataset.epoch[-1]])

    npix = hp.nside2npix(IDEX_HEALPIX_NSIDE)
    np.testing.assert_array_equal(pset.data.counts.shape, (npix,))
