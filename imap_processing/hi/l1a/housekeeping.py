"""Unpack IMAP-Hi housekeeping data."""
import collections
import dataclasses

import xarray as xr
from space_packet_parser.parser import Packet

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time
from imap_processing.hi.hi_cdf_attrs import (
    hi_hk_l1a_attrs,
    hi_hk_l1a_metadata_attrs,
)


def create_dataset(packets: list[Packet]) -> xr.Dataset:
    """Create dataset for each metadata field.

    Parameters
    ----------
    packets : list[Packet]
        packet list

    Returns
    -------
    xr.dataset
        dataset with all metadata field data in xr.DataArray
    """
    metadata_arrays = collections.defaultdict(list)
    description_dict = {}

    for data_packet in packets:
        # Add metadata to array
        for key, value in (data_packet.header | data_packet.data).items():
            # convert key to lower case to match SPDF requirement
            data_key = key.lower()
            metadata_arrays[data_key].append(value.raw_value)
            # description should be same for all packets
            description_dict[data_key] = (
                value.long_description or value.short_description
            )

    epoch_converted_time = [
        calc_start_time(sc_time) for sc_time in metadata_arrays["ccsds_met"]
    ]

    epoch_time = xr.DataArray(
        epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch_time},
        attrs=hi_hk_l1a_attrs.output(),
    )

    # create xarray dataset for each metadata field
    for key, value in metadata_arrays.items():
        if key == "instr_specific":
            # TODO: find out why this key has data
            # type as byte instead of uint or int like
            # other keys. Ask Vivek.
            continue
        # replace description and fieldname
        data_attrs = dataclasses.replace(
            hi_hk_l1a_metadata_attrs,
            catdesc=description_dict[key],
            fieldname=key,
            label_axis=key,
            depend_0="epoch",
        )
        dataset[key] = xr.DataArray(
            value,
            dims=["epoch"],
            attrs=data_attrs.output(),
        )

    return dataset
