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
    # TODO remove l2a and l2b?
    write_cdf(l2b_data)
"""

import logging

import astropy_healpix.healpy as hp
import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.idex.idex_constants import (
    IDEX_HEALPIX_NESTED,
    IDEX_HEALPIX_NSIDE,
    IDEX_POINTING_REFERENCE_FRAME,
)

logger = logging.getLogger(__name__)


def idex_l2c(l1b_datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    Will process IDEX l1b data to create l2c data products.

    Parameters
    ----------
    l1b_datasets : list[xarray.Dataset]
        IDEX L1b datasets.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L2C processing on datasets:"
        f" {[ds.attrs['Logical_source'] for ds in l1b_datasets]}"
    )

    # create the attribute manager for this data level
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs(instrument="idex")
    idex_attrs.add_instrument_variable_attrs("idex", "l2c")
    # Initialize the HealpixSkyMap and Rectangular object
    healpix_skymap = ena_maps.HealpixSkyMap(
        nside=IDEX_HEALPIX_NSIDE,
        nested=IDEX_HEALPIX_NESTED,
        spice_frame=IDEX_POINTING_REFERENCE_FRAME,
    )
    # Create raw dust count psets and push values to the map for each l1b dataset.
    for ds in l1b_datasets:
        pset = idex_pset(ds)

        # TODO exposure time
        # TODO rectangular map - code for this is coming shortly
        # TODO pull vs push?
        healpix_skymap.project_pset_values_to_map(pset, value_keys=["counts"])

    healpix_map_dataset = healpix_skymap.to_dataset()
    healpix_map_dataset.attrs = idex_attrs.get_global_attributes("imap_idex_l2c_sci")
    healpix_map_dataset["counts"].attrs = idex_attrs.get_variable_attributes("counts")
    healpix_map_dataset["healpix_index"].attrs = idex_attrs.get_variable_attributes(
        "healpix_index"
    )
    # Add attributes related to the map
    # Always add the following attributes to the map

    healpix_map_dataset.attrs.update(
        {
            "Sky_tiling_type": ena_maps.SkyTilingType.HEALPIX.value,
            "HEALPix_nside": IDEX_HEALPIX_NSIDE,
            "HEALPix_nest": IDEX_HEALPIX_NESTED,
            "Spice_reference_frame": IDEX_POINTING_REFERENCE_FRAME,
        }
    )

    logger.info("IDEX L2C science data processing completed.")
    return healpix_map_dataset


def idex_pset(
    l1b_dataset: xr.Dataset,
    nside: int = 8,
    nested: bool = False,
) -> ena_maps.IDEXPointingSet:
    """
    Create an IDEX pointing set object out of an l1b dataset.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        IDEX L2b dataset.
    nside : int
        Healpix nside parameter.
    nested : bool
        Healpix nested parameter.

    Returns
    -------
    pset : IDEXPointingSet
        IDEX pointing set object.
    """
    # For ISTP, epoch should be the center of the time bin.
    epoch_da = xr.DataArray(
        [np.mean(l1b_dataset["epoch"].data[[0, -1]]).astype(np.int64)],
        name="epoch",
        dims=["epoch"],
    )

    longitude = l1b_dataset["longitude"].copy()
    latitude = l1b_dataset["latitude"].copy()

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
    counds_da = xr.DataArray(
        counts,
        name="counts",
        dims=CoordNames.HEALPIX_INDEX.value,
    )
    l2c_dataset = xr.Dataset(
        coords={"healpix_index": healpix, "epoch": epoch_da},
        data_vars={"counts": counds_da},
    )

    return ena_maps.IDEXPointingSet(l2c_dataset, IDEX_POINTING_REFERENCE_FRAME)
