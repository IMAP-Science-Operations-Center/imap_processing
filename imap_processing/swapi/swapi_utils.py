"""Various classes and functions used throughout SWAPI processing.

This module contains utility classes and functions that are used by various
other SWAPI processing modules.
"""

import collections
from enum import IntEnum

import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates


class SWAPIAPID(IntEnum):
    """Create ENUM for apid.

    Parameters
    ----------
    IntEnum : IntEnum.
    """

    SWP_HK = 1184
    SWP_SCI = 1188
    SWP_AUT = 1192


class SWAPIMODE(IntEnum):
    """Create ENUM for MODE.

    Parameters
    ----------
    IntEnum : IntEnum.
    """

    LVENG = 0
    LVSCI = 1
    HVENG = 2
    HVSCI = 3


def add_metadata_to_array(data_packet, metadata_arrays):
    """Add metadata to the metadata_arrays.

    Parameters
    ----------
    data_packet : space_packet_parser.parser.Packet
        SWE data packet
    metadata_arrays : dict
        metadata arrays.
    """
    for key, value in (data_packet.header | data_packet.data).items():
        metadata_arrays.setdefault(key, []).append(value.raw_value)


def create_dataset(packets):
    """Create dataset for each metadata field.

    Parameters
    ----------
    packets : list
        packet list

    Returns
    -------
    xarray.dataset
        dataset with all metadata field data in xr.DataArray.
    """
    metadata_arrays = collections.defaultdict(list)

    for data_packet in packets:
        add_metadata_to_array(data_packet, metadata_arrays)

    epoch_time = xr.DataArray(
        metadata_arrays["SHCOARSE"],
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    data_vars = {
        key.lower(): xr.DataArray(value, dims=["epoch"])
        for key, value in metadata_arrays.items()
        if key != "SHCOARSE"
    }

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={"epoch": epoch_time},
    )

    return dataset
