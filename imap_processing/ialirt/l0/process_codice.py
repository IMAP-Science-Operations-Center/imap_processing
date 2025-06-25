"""Functions to support I-ALiRT CoDICE processing."""

import logging
from decimal import Decimal
from typing import Any

import xarray as xr

logger = logging.getLogger(__name__)

FILLVAL_FLOAT32 = Decimal(str(-1.0e31))


def process_codice(
    dataset: xr.Dataset,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Create final data products.

    Parameters
    ----------
    dataset : xr.Dataset
        Decommed L0 data.

    Returns
    -------
    codice_data : tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        Dictionary of final data product.

    Notes
    -----
    This function is incomplete and will need to be updated to include the
    necessary calculations and data products.
    - Calculate rates (assume 4 minutes per group)
    - Calculate L2 CoDICE pseudodensities (pg 37 of Algorithm Document)
    - Calculate the public data products
    """
    # For I-ALiRT SIT, the test data being used has all zeros and thus no
    # groups can be found, thus there is no data to process
    # TODO: Once I-ALiRT test data is acquired that actually has data in it,
    #       this can be turned back on
    # codicelo_data = create_ialirt_dataset(CODICEAPID.COD_LO_IAL, dataset)
    # codicehi_data = create_ialirt_dataset(CODICEAPID.COD_HI_IAL, dataset)

    # TODO: calculate rates
    #       This will be done in codice.codice_l1b

    # TODO: calculate L2 CoDICE pseudodensities
    #       This will be done in codice.codice_l2

    # TODO: calculate the public data products
    #       This will be done in this module

    # Create mock dataset for I-ALiRT SIT
    # TODO: Once I-ALiRT test data is acquired that actually has data in it,
    #       we should be able to properly populate the I-ALiRT data, but for
    #       now, just create lists of dicts.
    cod_lo_data: list[dict[str, Any]] = []
    cod_hi_data: list[dict[str, Any]] = []

    return cod_lo_data, cod_hi_data
