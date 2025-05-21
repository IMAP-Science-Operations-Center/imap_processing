"""Functions to support I-ALiRT CoDICE Lo processing."""

import logging

import xarray as xr

from imap_processing.codice.codice_l1a import create_ialirt_dataset

logger = logging.getLogger(__name__)


def process_codicelo(dataset: xr.Dataset) -> list[dict]:
    """
    Create final data products.

    Parameters
    ----------
    dataset : xr.Dataset
        Decommed L0 data.

    Returns
    -------
    codicelo_data : list[dict]
        Dictionary of final data product.

    Notes
    -----
    This function is incomplete and will need to be updated to include the
    necessary calculations and data products.
    - Calculate rates (assume 4 minutes per group)
    - Calculate L2 CoDICE pseudodensities (pg 37 of Algorithm Document)
    - Calculate the public data products
    """
    apid = dataset.pkt_apid.data[0]
    codicelo_data = create_ialirt_dataset(apid, dataset)

    # TODO: calculate rates
    # TODO: calculate L2 CoDICE pseudodensities
    # TODO: calculate the public data products

    return codicelo_data
