"""Tests the L2c processing for IDEX data"""

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import write_cdf
from imap_processing.idex.idex_constants import (
    IDEX_HEALPIX_NESTED,
    IDEX_HEALPIX_NSIDE,
    IDEX_POINTING_REFERENCE_FRAME,
)
from imap_processing.idex.idex_l2c import idex_l2c, idex_pset


@pytest.fixture
def l2c_dataset(l1b_dataset: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """

    return idex_l2c(l1b_dataset)


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
    # in the l1b_dataset
    np.testing.assert_allclose(l2c_dataset["counts"].sum(), len(l1b_dataset.epoch))
    assert l2c_dataset.dims == {
        "healpix_index": hp.nside2npix(IDEX_HEALPIX_NSIDE),
        "epoch": 1,
    }

    # Assert attributes are present
    assert l2c_dataset.attrs["sky_tiling_type"] == "Healpix"
    assert l2c_dataset.attrs["HEALPix_nside"] == IDEX_HEALPIX_NSIDE
    assert l2c_dataset.attrs["HEALPix_nest"] == IDEX_HEALPIX_NESTED
    assert l2c_dataset.attrs["spice_reference_frame"] == IDEX_POINTING_REFERENCE_FRAME
    # Check the attributes of the dataset by writing to a CDF file
    write_cdf(l2c_dataset)


def test_idex_pset(l1b_dataset: xr.Dataset):
    """Test for idex_pset function"""
    pset = idex_pset(l1b_dataset)

    assert pset.epoch == np.mean([l1b_dataset.epoch[0], l1b_dataset.epoch[-1]])

    npix = hp.nside2npix(IDEX_HEALPIX_NSIDE)
    np.testing.assert_array_equal(pset.counts.shape, (npix,))
