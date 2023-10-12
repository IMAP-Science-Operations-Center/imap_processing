import collections
from enum import IntEnum

import xarray as xr


class SWAPIAPID(IntEnum):
    """Create ENUM for apid.

    Parameters
    ----------
    IntEnum : IntEnum.
    """

    SWP_HK = 1184
    SWP_SCI = 1188
    SWP_AUT = 1192


def add_metadata_to_array(data_packet, metadata_arrays):
    """Add metadata to the metadata_arrays.

    Parameters
    ----------
    data_packet : space_packet_parser.parser.Packet
        SWE data packet
    metadata_arrays : dict
        metadata arrays.
    """
    for key, value in data_packet.header.items():
        metadata_arrays.setdefault(key, []).append(value.raw_value)
    for key, value in data_packet.data.items():
        metadata_arrays.setdefault(key, []).append(value.raw_value)


def create_dataset(packets):
    """Create dataset for each metadata field.

    Parameters
    ----------
    packets : list
        packet list

    Returns
    -------
    xr.dataset
        dataset with all metadata field data in xr.DataArray.
    """
    metadata_arrays = collections.defaultdict(list)

    for data_packet in packets:
        add_metadata_to_array(data_packet, metadata_arrays)

    epoch_time = xr.DataArray(
        metadata_arrays["SHCOARSE"],
        name="Epoch",
        dims=["Epoch"],
        attrs=dict(
            description="Mission elapsed time",
            units="seconds since start of the mission",
        ),
    )

    dataset = xr.Dataset(
        coords={"Epoch": epoch_time},
    )

    # create xarray dataset for each metadata field
    for key, value in metadata_arrays.items():
        if key == "SHCOARSE":
            continue
        dataset[key] = xr.DataArray(
            value,
            dims=["Epoch"],
        )
    return dataset
