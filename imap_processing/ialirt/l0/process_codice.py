"""Functions to support I-ALiRT CoDICE processing."""

import logging
from decimal import Decimal
from typing import Any

import xarray as xr

from imap_processing.codice import constants
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_ttj2000ns, met_to_utc

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
    codice_data : list[dict]
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
    #       now, just create lists of dicts with FILLVALs
    cod_lo_data = []
    cod_hi_data = []

    for epoch in range(len(dataset.epoch.data)):
        sc_sclk_sec = dataset.sc_sclk_sec.data[epoch]
        sc_sclk_sub_sec = dataset.sc_sclk_sub_sec.data[epoch]
        met = calculate_time(sc_sclk_sec, sc_sclk_sub_sec, 256)
        utc = met_to_utc(met).split(".")[0]
        ttj2000ns = int(met_to_ttj2000ns(met))

        epoch_data = {
            "apid": int(dataset.pkt_apid[epoch].data),
            "met": met,
            "met_to_utc": utc,
            "ttj2000ns": ttj2000ns,
        }

        # Add in CoDICE-Lo specific data
        cod_lo_epoch_data = epoch_data.copy()
        for field in constants.CODICE_LO_IAL_DATA_FIELDS:
            cod_lo_epoch_data[f"codicelo_{field}"] = FILLVAL_FLOAT32
        cod_lo_data.append(cod_lo_epoch_data)

        # Add in CoDICE-Hi specific data
        cod_hi_epoch_data = epoch_data.copy()
        for field in constants.CODICE_HI_IAL_DATA_FIELDS:
            cod_hi_epoch_data[f"codicehi_{field}"] = FILLVAL_FLOAT32
        cod_hi_data.append(cod_hi_epoch_data)

    return cod_lo_data, cod_hi_data
