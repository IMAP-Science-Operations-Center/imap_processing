"""Processing function for Lo star sensor data."""

import logging

import numpy as np
import xarray as xr

from imap_processing.lo.l0.utils.bit_decompression import (
    DECOMPRESSION_TABLES,
    Decompress,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_star_sensor(ds: xr.Dataset) -> xr.Dataset:
    """
    Process Lo star sensor data.

    Parameters
    ----------
    ds : xr.Dataset
        The packet dataset containing Lo star sensor data.

    Returns
    -------
    xr.Dataset
        Processed dataset with a decompressed data field.
    """
    # Make one long flat buffer
    # This assumes that all data_compressed entries are of the same length
    # but allows for only one frombuffer call
    buffer = b"".join(ds["data_compressed"].values)
    data = np.frombuffer(buffer, dtype=np.uint8).reshape(-1, 720)

    # Decompress from 8 -> 12 bits using the decompression tables
    decompression = DECOMPRESSION_TABLES[Decompress.DECOMPRESS8TO12].astype(np.uint16)
    # Use the mean value column (2)
    data = decompression[data, 2]

    # There is already a variable called "count" in the dataset that
    # came with the packet
    ds["data_index"] = xr.DataArray(np.arange(720), dims="data_index")
    ds["data"] = xr.DataArray(data, dims=("epoch", "data_index"))
    # Remove the original compressed data field
    ds = ds.drop_vars("data_compressed")
    return ds
