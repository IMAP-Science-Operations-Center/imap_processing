"""Unpack IMAP-Hi housekeeping data."""

import xarray as xr

from imap_processing.hi.hi_cdf_attrs import (
    hi_hk_l1a_attrs,
)


def process_housekeeping(dataset: xr.Dataset) -> xr.Dataset:
    """
    Create dataset for each metadata field.

    Parameters
    ----------
    dataset : xarray.Dataset
        Packet input dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all metadata field data in xr.DataArray.
    """
    # Add datalevel attrs
    dataset.attrs.update(hi_hk_l1a_attrs.output())
    return dataset
