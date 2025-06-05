"""
Perform IDEX L2c Processing.

Examples
--------
.. code-block:: python
    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b
    from imap_processing.idex.idex_l2a import idex_l2a
    from imap_processing.idex.idex_l2b import idex_l2b
    from imap_processing.cdf.utils import write_cdf

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file)
    l1b_data = idex_l1b(l1a_data)
    l2a_data = idex_l2a(l1b_data)
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
    IDEX_EVENT_REFERENCE_FRAME,
    IDEX_HEALPIX_NESTED,
    IDEX_HEALPIX_NSIDE,
    IDEX_SPACING_DEG,
)
from imap_processing.idex.idex_utils import get_idex_attrs

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
    # create the attribute manager for this data level
    idex_attrs = get_idex_attrs("l2c")
    # Epoch should be the start of the collection period.
    # TODO should epoch be start of sci acquisition?
    epoch = xr.DataArray(
        l2b_dataset["epoch"].data[0:1].astype(np.int64),
        name="epoch",
        dims=["epoch"],
        attrs=idex_attrs.get_variable_attributes(
            "epoch_collection_set", check_schema=False
        ),
    )
    l2c_healpix_dataset = idex_healpix_map(l2b_dataset, epoch, idex_attrs)
    l2c_rectangular_dataset = idex_rectangular_map(l2b_dataset, epoch, idex_attrs)

    # TODO exposure time
    logger.info("IDEX L2C science data processing completed.")
    return [l2c_healpix_dataset, l2c_rectangular_dataset]


def idex_healpix_map(
    l1b_dataset: xr.Dataset,
    epoch_da: xr.DataArray,
    idex_attrs: ImapCdfAttributes,
    nside: int = IDEX_HEALPIX_NSIDE,
    nested: bool = IDEX_HEALPIX_NESTED,
) -> xr.Dataset:
    """
    Create a healpix map out of a l1b dataset.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        IDEX L2b dataset.
    epoch_da : xarray.DataArray
        Epoch data array of the collection. Size: (1,).
    idex_attrs : ImapCdfAttributes
        The attribute manager for this data level.
    nside : int
        Healpix nside parameter.
    nested : bool
        Healpix nested parameter.

    Returns
    -------
    map : xarray.Dataset
        Spatially binned dust counts in a healpix map format.
    """
    longitude = l1b_dataset["longitude"]
    latitude = l1b_dataset["latitude"]

    # Get the healpix indices
    hpix_idx = hp.ang2pix(
        nside, nest=nested, lonlat=True, theta=longitude, phi=latitude
    )

    n_pix = hp.nside2npix(nside)
    healpix = xr.DataArray(
        np.arange(n_pix),
        name=CoordNames.HEALPIX_INDEX.value,
        dims=CoordNames.HEALPIX_INDEX.value,
        attrs=idex_attrs.get_variable_attributes("pixel_index", check_schema=False),
    )

    # Create a histogram of the raw dust event counts for each pixel
    counts = np.histogram(hpix_idx, bins=n_pix, range=(0, n_pix))[0]
    # Add epoch dimension
    counts_da = xr.DataArray(
        counts[np.newaxis, :].astype(np.int64),
        name="counts",
        dims=("epoch", CoordNames.HEALPIX_INDEX.value),
        attrs=idex_attrs.get_variable_attributes("healpix_counts"),
    )
    pixel_label = xr.DataArray(
        healpix.astype(str),
        name="pixel_label",
        dims="pixel_index",
        attrs=idex_attrs.get_variable_attributes("pixel_label", check_schema=False),
    )
    l2c_dataset = xr.Dataset(
        coords={CoordNames.HEALPIX_INDEX.value: healpix, "epoch": epoch_da},
        data_vars={
            "counts": counts_da,
            "longitude": longitude,
            "latitude": latitude,
            "pixel_label": pixel_label,
        },
    )
    map_attrs = {
        "Sky_tiling_type": SkyTilingType.HEALPIX.value,
        "HEALPix_nside": str(nside),
        "HEALPix_nest": str(nested),
        "Spice_reference_frame": IDEX_EVENT_REFERENCE_FRAME.name,
        "num_points": str(n_pix),
    } | idex_attrs.get_global_attributes("imap_idex_l2c_sci-healpix")
    l2c_dataset.attrs.update(map_attrs)

    return l2c_dataset


def idex_rectangular_map(
    l1b_dataset: xr.Dataset,
    epoch_da: xr.DataArray,
    idex_attrs: ImapCdfAttributes,
    spacing_deg: int = IDEX_SPACING_DEG,
) -> xr.Dataset:
    """
    Create a rectangular map out of a l1b dataset.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        IDEX L2b dataset.
    epoch_da : xarray.DataArray
        Epoch data array of the collection. Size: (1,).
    idex_attrs : ImapCdfAttributes
        The attribute manager for this data level.
    spacing_deg : int
        The spacing in degrees for the rectangular grid.

    Returns
    -------
    map : xarray.Dataset
        Spatially binned dust counts in a rectangular map format.
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
        counts[np.newaxis, :, :].astype(np.int64),
        name="counts",
        dims=("epoch", "rectangular_lon_pixel", "rectangular_lat_pixel"),
        attrs=idex_attrs.get_variable_attributes("rectangular_counts"),
    )
    rec_lon_pixels = xr.DataArray(
        name="rectangular_lon_pixel",
        data=grid.az_bin_midpoints,
        dims="rectangular_lon_pixel",
        attrs=idex_attrs.get_variable_attributes(
            "rectangular_lon_pixel", check_schema=False
        ),
    )
    rec_lat_pixels = xr.DataArray(
        name="rectangular_lat_pixel",
        data=grid.el_bin_midpoints,
        dims="rectangular_lat_pixel",
        attrs=idex_attrs.get_variable_attributes(
            "rectangular_lat_pixel", check_schema=False
        ),
    )

    l2c_dataset = xr.Dataset(
        coords={
            "epoch": epoch_da,
            "rectangular_lon_pixel": rec_lon_pixels,
            "rectangular_lat_pixel": rec_lat_pixels,
        },
        data_vars={
            "counts": counts_da,
            "longitude": longitude_wrapped,
            "latitude": latitude,
            "rectangular_lon_pixel_label": rec_lon_pixels.astype(str),
            "rectangular_lat_pixel_label": rec_lat_pixels.astype(str),
        },
    )
    l2c_dataset[
        "rectangular_lon_pixel_label"
    ].attrs = idex_attrs.get_variable_attributes(
        "rectangular_lon_pixel_label", check_schema=False
    )
    l2c_dataset[
        "rectangular_lat_pixel_label"
    ].attrs = idex_attrs.get_variable_attributes(
        "rectangular_lat_pixel_label", check_schema=False
    )
    map_attrs = {
        "sky_tiling_type": SkyTilingType.RECTANGULAR.value,
        "Spacing_degrees": str(spacing_deg),
        "Spice_reference_frame": IDEX_EVENT_REFERENCE_FRAME.name,
        "num_points": str(counts.size),
    } | idex_attrs.get_global_attributes("imap_idex_l2c_sci-rectangular")

    l2c_dataset.attrs.update(map_attrs)
    return l2c_dataset
