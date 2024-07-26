"""Unpack IMAP-Hi housekeeping data."""

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


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
    # Load the CDF attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    # Add datalevel attrs
    dataset.attrs.update(attr_mgr.get_global_attributes("imap_hi_l1a_hk_attrs"))
    return dataset
