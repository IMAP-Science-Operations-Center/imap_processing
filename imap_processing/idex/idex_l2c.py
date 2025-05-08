"""
Perform IDEX L2c Processing.

Examples
--------
.. code-block:: python
    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b
    from imap_processing.idex.idex_l1b import idex_l2a
    from imap_processing.idex.idex_l1b import idex_l2b

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file)
    l1b_data = idex_l1b(l1a_data)
    l1a_data = idex_l2a(l1b_data)
    l2b_data = idex_l2b(l2a_data)
    write_cdf(l2b_data)
"""

import logging

import astropy_healpix.healpy as hp
import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps.ena_maps import SkyTilingType
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.spatial_utils import AzElSkyGrid
from imap_processing.idex.idex_constants import (
    IDEX_HEALPIX_NESTED,
    IDEX_HEALPIX_NSIDE,
    IDEX_POINTING_REFERENCE_FRAME,
    IDEX_SPACING_DEG,
)

logger = logging.getLogger(__name__)


def idex_l2c(l2b_dataset: xr.Dataset) -> list[xr.Dataset]:
    """
    Will process IDEX l2b data to create l2c data products.

    Parameters
    ----------
    l2b_dataset : xarray.Dataset
        IDEX L2b dataset.

    Returns
    -------
    l2b_dataset : list[xarray.Dataset]
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L2C processing on datasets: "
        f"{l2b_dataset.attrs['Logical_source']}"
    )
    # For ISTP, epoch should be the center of the time bin.
    epoch = xr.DataArray(
        [np.mean(l2b_dataset["epoch"].data[[0, -1]]).astype(np.int64)],
        name="epoch",
        dims=["epoch"],
    )
    l2c_healpix_dataset = idex_healpix_pset(l2b_dataset, epoch)
    l2c_rectangular_dataset = idex_rectangular_pset(l2b_dataset, epoch)
    # create the attribute manager for this data level
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs(instrument="idex")
    idex_attrs.add_instrument_variable_attrs("idex", "l2c")

    # TODO exposure time
    l2c_healpix_dataset.attrs.update(
        idex_attrs.get_global_attributes("imap_idex_l2c_sci-healpix")
    )
    l2c_healpix_dataset["healpix_counts"].attrs = idex_attrs.get_variable_attributes(
        "healpix_counts"
    )
    l2c_healpix_dataset["epoch"].attrs = idex_attrs.get_variable_attributes("epoch")
    l2c_healpix_dataset["pixel_index"].attrs = idex_attrs.get_variable_attributes(
        "pixel_index"
    )

    l2c_rectangular_dataset.attrs.update(
        idex_attrs.get_global_attributes("imap_idex_l2c_sci-rectangular")
    )
    l2c_rectangular_dataset[
        "rectangular_counts"
    ].attrs = idex_attrs.get_variable_attributes("rectangular_counts")
    l2c_rectangular_dataset["epoch"].attrs = idex_attrs.get_variable_attributes("epoch")
    l2c_rectangular_dataset[
        "rectangular_lat_pixel"
    ].attrs = idex_attrs.get_variable_attributes("rectangular_lat_pixel")
    l2c_rectangular_dataset[
        "rectangular_lon_pixel"
    ].attrs = idex_attrs.get_variable_attributes("rectangular_lon_pixel")
    l2c_rectangular_dataset[
        "rectangular_lat_pixel_label"
    ].attrs = idex_attrs.get_variable_attributes("rectangular_lat_pixel_label")
    l2c_rectangular_dataset[
        "rectangular_lon_pixel_label"
    ].attrs = idex_attrs.get_variable_attributes("rectangular_lon_pixel_label")

    logger.info("IDEX L2C science data processing completed.")
    return [l2c_healpix_dataset, l2c_rectangular_dataset]


def idex_healpix_pset(
    l1b_dataset: xr.Dataset,
    epoch_da: xr.DataArray,
    nside: int = IDEX_HEALPIX_NSIDE,
    nested: bool = IDEX_HEALPIX_NESTED,
) -> xr.Dataset:
    """
    Create a healpix IDEX pointing set out of an l1b dataset.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        IDEX L2b dataset.
    epoch_da : xarray.DataArray
        Epoch data array of size (1,).
    nside : int
        Healpix nside parameter.
    nested : bool
        Healpix nested parameter.

    Returns
    -------
    pset : xarray.Dataset
        IDEX pointing set dataset.
    """
    longitude = l1b_dataset["longitude"].copy()
    latitude = l1b_dataset["latitude"].copy()

    # Get the healpix indices
    hpix_idx = hp.ang2pix(
        nside, nest=nested, lonlat=True, theta=longitude, phi=latitude
    )

    n_pix = hp.nside2npix(nside)
    healpix = xr.DataArray(
        np.arange(n_pix),
        name=CoordNames.HEALPIX_INDEX.value,
        dims=CoordNames.HEALPIX_INDEX.value,
    )

    # Create a histogram of the raw dust event counts for each pixel
    counts = np.histogram(hpix_idx, bins=n_pix, range=(0, n_pix))[0]
    counts_da = xr.DataArray(
        counts,
        name="healpix_counts",
        dims=CoordNames.HEALPIX_INDEX.value,
    )
    l2c_dataset = xr.Dataset(
        coords={CoordNames.HEALPIX_INDEX.value: healpix, "epoch": epoch_da},
        data_vars={
            "healpix_counts": counts_da,
            "longitude": longitude,
            "latitude": latitude,
        },
    )
    pset_attrs = {
        "sky_tiling_type": SkyTilingType.HEALPIX.value,
        "HEALPix_nside": nside,
        "HEALPix_nest": nested,
        "spice_reference_frame": IDEX_POINTING_REFERENCE_FRAME,
        "num_points": n_pix,
    }
    l2c_dataset.attrs.update(pset_attrs)

    return l2c_dataset


def idex_rectangular_pset(
    l1b_dataset: xr.Dataset, epoch_da: xr.DataArray, spacing_deg: int = IDEX_SPACING_DEG
) -> xr.Dataset:
    """
    Create a rectangular IDEX pointing set object out of a l1b dataset.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        IDEX L2b dataset.
    epoch_da : xarray.DataArray
        Epoch data array of size (1,).
    spacing_deg : int
        The spacing in degrees for the rectangular grid.

    Returns
    -------
    pset : xarray.Dataset
        IDEX pointing set dataset.
    """
    # Get the rectangular grid with the specified spacing
    grid = AzElSkyGrid(spacing_deg)
    # Make sure longitude values are in the range [0, 360)
    longitude_wrapped = np.mod(l1b_dataset["longitude"], 360)
    latitude = l1b_dataset["latitude"]
    # Create a 2d histogram of the raw dust event counts for each pixel using the grid
    # bin edges
    counts, _, _ = np.histogram2d(
        longitude_wrapped, latitude, bins=[grid.az_bin_edges, grid.el_bin_edges]
    )
    counts_da = xr.DataArray(
        counts,
        name="rectangular_counts",
        dims=("rectangular_lon_pixel", "rectangular_lat_pixel"),
    )
    rec_lon_pixels = xr.DataArray(
        np.arange(grid.grid_shape[0]), dims="rectangular_lon_pixel"
    )
    rec_lat_pixels = xr.DataArray(
        np.arange(grid.grid_shape[1]), dims="rectangular_lat_pixel"
    )
    l2c_dataset = xr.Dataset(
        coords={
            "epoch": epoch_da,
            "rectangular_lon_pixel": rec_lon_pixels,
            "rectangular_lat_pixel": rec_lat_pixels,
        },
        data_vars={
            "rectangular_counts": counts_da,
            "longitude": longitude_wrapped,
            "latitude": latitude,
            "rectangular_lon_pixel_label": rec_lon_pixels.astype(str),
            "rectangular_lat_pixel_label": rec_lat_pixels.astype(str),
        },
    )
    pset_attrs = {
        "sky_tiling_type": SkyTilingType.RECTANGULAR.value,
        "spacing_degree": spacing_deg,
        "spice_reference_frame": IDEX_POINTING_REFERENCE_FRAME,
        "num_points": counts.size,
    }
    l2c_dataset.attrs.update(pset_attrs)
    return l2c_dataset
