"""Functions to support I-ALiRT CoDICE processing."""

import logging
from decimal import Decimal
from typing import Any

import numpy as np
import xarray as xr

from imap_processing.ialirt.utils.grouping import find_groups

logger = logging.getLogger(__name__)

FILLVAL_UINT8 = 255
FILLVAL_FLOAT32 = Decimal(str(-1.0e31))
COD_LO_COUNTER = 232
COD_HI_COUNTER = 197
COD_LO_RANGE = range(0, 15)
COD_HI_RANGE = range(0, 5)


def concatenate_bytes(
    grouped_cod_lo_data: xr.Dataset, group: int, sensor: str
) -> bytearray:
    """
    Concatenate all cod_lo_data fields for a specific group into a single bytearray.

    Parameters
    ----------
    grouped_cod_lo_data : xr.Dataset
        The grouped CoDICE-Lo dataset containing cod_lo_data_XX variables.
    group : int
        The group number to extract.
    sensor : str
        The sensor type, either 'lo' or 'hi'.

    Returns
    -------
    bytearray
        The concatenated data stream for the selected group.
    """
    current_data_stream = bytearray()
    group_mask = (grouped_cod_lo_data["group"] == group).values

    cod_ranges = {
        "lo": COD_LO_RANGE,
        "hi": COD_HI_RANGE,
    }

    # Loop through all data fields.
    for field in cod_ranges[sensor]:
        data_array = grouped_cod_lo_data[f"cod_{sensor}_data_{field:02}"].values[
            group_mask
        ]

        # Convert each value to uint8 and extend the byte stream
        current_data_stream.extend(np.uint8(data_array).tobytes())

    return current_data_stream


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
    grouped_cod_lo_data = find_groups(
        dataset, (0, COD_LO_COUNTER), "cod_lo_counter", "cod_lo_acq"
    )
    grouped_cod_hi_data = find_groups(
        dataset, (0, COD_HI_COUNTER), "cod_hi_counter", "cod_hi_acq"
    )
    unique_cod_lo_groups = np.unique(grouped_cod_lo_data["group"])
    unique_cod_hi_groups = np.unique(grouped_cod_hi_data["group"])

    for group in unique_cod_lo_groups:
        cod_lo_data_stream = concatenate_bytes(grouped_cod_lo_data, group, "lo")

    for group in unique_cod_hi_groups:
        cod_hi_data_stream = concatenate_bytes(grouped_cod_hi_data, group, "lo")

        print("hi")

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
