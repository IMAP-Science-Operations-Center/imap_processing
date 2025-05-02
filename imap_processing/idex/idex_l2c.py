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
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.idex.idex_constants import (
    IDEX_POINTING_REFERENCE_FRAME,
)

logger = logging.getLogger(__name__)


def idex_l2c(l2b_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process IDEX l2b data to create l2c data products.

    Parameters
    ----------
    l2b_dataset : xarray.Dataset
        IDEX L2b dataset.

    Returns
    -------
    l2b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L2C processing on datasets: "
        f"{l2b_dataset.attrs['Logical_source']}"
    )

    # create the attribute manager for this data level
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs(instrument="idex")
    idex_attrs.add_instrument_variable_attrs("idex", "l2c")

    # Create a raw dust count pset
    pset = idex_pset(l2b_dataset)
    pset_dataset = pset.to_dataset()
    # TODO exposure time
    # TODO rectangular map
    pset_dataset.attrs.update(idex_attrs.get_global_attributes("imap_idex_l2c_sci"))
    pset_dataset["counts"].attrs = idex_attrs.get_variable_attributes("counts")
    pset_dataset["epoch"].attrs = idex_attrs.get_variable_attributes("epoch")
    pset_dataset["healpix_index"].attrs = idex_attrs.get_variable_attributes(
        "healpix_index"
    )
    logger.info("IDEX L2C science data processing completed.")
    return pset_dataset


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
