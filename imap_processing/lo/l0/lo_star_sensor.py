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


def unpack_star_sensor(ds: xr.Dataset) -> xr.Dataset:
    """
    Unpack Lo star sensor data.

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

    # There is already a variable called "count" in the dataset that
    # came with the packet
    ds["data_index"] = xr.DataArray(np.arange(720), dims="data_index")
    ds["data"] = xr.DataArray(data, dims=("epoch", "data_index"))
    # Remove the original compressed data field
    ds = ds.drop_vars("data_compressed")
    return ds


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
        Processed dataset with unpacked and decompressed data.
    """
    # Decompress from 8 -> 12 bits using the decompression tables
    decompression = DECOMPRESSION_TABLES[Decompress.DECOMPRESS8TO12].astype(np.uint16)

    # Use the mean value column (2)
    ds["data"].values = decompression[ds["data"].values, 2]

    # We don't want the original l1a ccsds variables, only data
    vars_to_drop = [x for x in ds.data_vars if x != "data"]
    ds = ds.drop_vars(vars_to_drop)

    return ds
