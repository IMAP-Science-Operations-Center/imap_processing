"""Functions to support I-ALiRT CoDICE processing."""

import logging
from decimal import Decimal
from typing import Any

import numpy as np
import xarray as xr

from imap_processing.codice import decompress
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes

logger = logging.getLogger(__name__)

FILLVAL_UINT8 = 255
FILLVAL_FLOAT32 = Decimal(str(-1.0e31))
COD_LO_COUNTER = 232
COD_HI_COUNTER = 197
COD_LO_RANGE = range(0, 15)
COD_HI_RANGE = range(0, 5)


def concatenate_bytes(grouped_data: xr.Dataset, group: int, sensor: str) -> bytearray:
    """
    Concatenate all data fields for a specific group into a single bytearray.

    Parameters
    ----------
    grouped_data : xr.Dataset
        The grouped CoDICE dataset containing cod_{sensor}_data_XX variables.
    group : int
        The group number to extract.
    sensor : str
        The sensor type, either 'lo' or 'hi'.

    Returns
    -------
    current_data_stream: bytearray
        The concatenated data stream for the selected group.
    """
    current_data_stream = bytearray()
    group_mask = (grouped_data["group"] == group).values

    cod_ranges = {
        "lo": COD_LO_RANGE,
        "hi": COD_HI_RANGE,
    }

    # Stack all cod_* fields into a 2D NumPy array [n_rows, n_fields]
    arrays = [
        grouped_data[f"cod_{sensor}_data_{field:02}"].values[group_mask]
        for field in cod_ranges[sensor]
    ]

    # Shape → (n_fields, n_rows)
    stacked = np.vstack(arrays)

    # Transpose to get (n_rows, n_fields), then flatten row-wise
    flattened = stacked.T.flatten()

    # Convert to bytes and extend the stream
    current_data_stream.extend(np.uint8(flattened).tobytes())

    return current_data_stream


def create_xarray_dataset(science_values, metadata_values, sensor) -> xr.Dataset:
    """
    Create an xarray Dataset from science and metadata values.

    Parameters
    ----------
    science_values : list[int]
        List of decompressed science values.
    metadata_values : dict[str, list[Any]]
        Dictionary of metadata values.
    sensor : str
        The sensor type, either 'lo' or 'hi'.

    Returns
    -------
    xr.Dataset
        The constructed xarray Dataset.
    """
    apid = {"lo": 1152, "hi": 1168}

    science_byte_values = []
    # Placeholder for epoch dimension that will not be used.
    epoch = np.arange(len(science_values))

    epoch_time = xr.DataArray(
        epoch,
        name="epoch",
        dims=["epoch"],
    )
    dataset = xr.Dataset(
        coords={"epoch": epoch_time},
    )

    for key in metadata_values.keys():
        data = np.array(metadata_values[key])
        dataset[key.lower()] = xr.DataArray(
            data,
            dims=["epoch"],
        )

    for i in range(len(science_values)):
        values = int(science_values[i], 2).to_bytes(
            len(science_values[i]) // 8, byteorder="big"
        )
        science_byte_values.append(values)

    dataset["data"] = xr.DataArray(
        science_byte_values,
        dims=["epoch"],
    )

    dataset["pkt_apid"] = xr.DataArray(
        np.array([apid[sensor]] * 9),
        dims=["epoch"],
    )

    return dataset


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

        # Decompress binary stream
        decompressed_data = decompress._apply_pack_24_bit(bytes(cod_lo_data_stream))

    for group in unique_cod_hi_groups:
        cod_hi_data_stream = concatenate_bytes(grouped_cod_hi_data, group, "hi")

        # Decompress binary stream
        decompressed_data = decompress._apply_lossy_a(bytes(cod_hi_data_stream))  # noqa

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
