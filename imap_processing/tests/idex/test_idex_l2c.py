"""Tests the L2c processing for IDEX data"""

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import write_cdf
from imap_processing.ena_maps.utils.spatial_utils import AzElSkyGrid
from imap_processing.idex.idex_constants import (
    IDEX_HEALPIX_NSIDE,
    IDEX_SPACING_DEG,
)
from imap_processing.idex.idex_l2c import (
    idex_healpix_map,
    idex_l2c,
    idex_rectangular_map,
)


@pytest.fixture
def l2c_datasets(l1b_dataset: xr.Dataset) -> list[xr.Dataset]:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : list[xr.Dataset]
        A list of ``xarray`` datasets containing the test data
    """

    return idex_l2c(l1b_dataset)


def test_l2c_attrs_and_vars(l2c_datasets: list[xr.Dataset], l1b_dataset: xr.Dataset):
    """Tests that the ``idex_l2b`` function generates datasets
    with the expected variables and attributes.

    Parameters
    ----------
    l2c_datasets : list[xr.Dataset]
        A ``xarray`` dataset containing the l2c test data.
    l1b_dataset
        A ``xarray`` dataset containing the l1b test data.
    """
    healpix_ds = l2c_datasets[0]
    rect_ds = l2c_datasets[1]
    assert healpix_ds.attrs["Logical_source"] == "imap_idex_l2c_healpix-pset-1week"
    assert rect_ds.attrs["Logical_source"] == "imap_idex_l2c_rectangular-pset-1week"
    # The total counts in the pset should be equal to the number of dust events
    # in the l1b_dataset
    np.testing.assert_allclose(
        healpix_ds["healpix_counts"].sum(), len(l1b_dataset.epoch)
    )
    np.testing.assert_allclose(
        rect_ds["rectangular_counts"].sum(), len(l1b_dataset.epoch)
    )
    assert healpix_ds.dims == {
        "pixel_index": hp.nside2npix(IDEX_HEALPIX_NSIDE),
        "epoch": 1,
    }

    assert rect_ds.dims == {
        "rectangular_lon_pixel": int(360 / IDEX_SPACING_DEG),
        "rectangular_lat_pixel": int(180 / IDEX_SPACING_DEG),
        "epoch": 1,
    }
    # Check the attributes of the dataset by writing to a CDF file
    write_cdf(healpix_ds)
    write_cdf(rect_ds)


def test_idex_healpix_map(l1b_dataset: xr.Dataset):
    """Test for idex_healpix_map function"""
    epoch = xr.DataArray(
        [np.mean(l1b_dataset["epoch"].data[[0, -1]]).astype(np.int64)],
        name="epoch",
        dims=["epoch"],
    )
    pset = idex_healpix_map(l1b_dataset, epoch)

    assert pset.epoch == np.mean([l1b_dataset.epoch[0], l1b_dataset.epoch[-1]])

    npix = hp.nside2npix(IDEX_HEALPIX_NSIDE)
    np.testing.assert_array_equal(pset.healpix_counts.shape, (npix,))


def test_idex_rectangular_map(l1b_dataset: xr.Dataset):
    """Test for idex_rectangular_map function"""
    epoch = xr.DataArray(
        [np.mean(l1b_dataset["epoch"].data[[0, -1]]).astype(np.int64)],
        name="epoch",
        dims=["epoch"],
    )
    pset = idex_rectangular_map(l1b_dataset, epoch)

    assert pset.epoch == np.mean([l1b_dataset.epoch[0], l1b_dataset.epoch[-1]])

    np.testing.assert_array_equal(
        pset.rectangular_counts.shape,
        (
            int(360 / IDEX_SPACING_DEG),
            int(180 / IDEX_SPACING_DEG),
        ),
    )

    expected_counts = np.zeros(
        (int(360 / IDEX_SPACING_DEG), int(180 / IDEX_SPACING_DEG))
    )
    grid = AzElSkyGrid(IDEX_SPACING_DEG)
    for lon, lat in zip(l1b_dataset["longitude"].data, l1b_dataset["latitude"].data):
        lon_wrapped = lon % 360
        az_indices = (
            np.digitize(
                lon_wrapped,
                grid.az_bin_edges,
            )
            - 1
        )
        el_indices = (
            np.digitize(
                lat,
                grid.el_bin_edges,
            )
            - 1
        )
        expected_counts[az_indices, el_indices] += 1

    np.testing.assert_array_equal(expected_counts, pset.rectangular_counts.data)
