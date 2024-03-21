"""Unpack IMAP-Hi housekeeping data."""
import collections
import dataclasses

import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time
from imap_processing.hi.hi_cdf_attrs import (
    hi_hk_l1a_attrs,
    hi_hk_l1a_metadata_attrs,
)


def create_dataset(packets):
    """Create dataset for each metadata field.

    Parameters
    ----------
    packets : list
        packet list

    Returns
    -------
    xr.dataset
        dataset with all metadata field data in xr.DataArray
    """
    metadata_arrays = collections.defaultdict(list)
    description_arrays = collections.defaultdict(list)

    for data_packet in packets:
        # Add metadata to array
        for key, value in data_packet.header.items():
            # convert key to lower case to match SPDF requirement
            data_key = key.lower()
            metadata_arrays.setdefault(data_key, []).append(value.raw_value)
            # add it once since description should be same for all packets
            if data_key not in description_arrays:
                description_arrays[data_key] = (
                    value.long_description
                    if value.long_description
                    else value.short_description
                )
        for key, value in data_packet.data.items():
            data_key = key.lower()
            metadata_arrays.setdefault(data_key, []).append(value.raw_value)
            # add it once since description should be same for all packets
            if data_key not in description_arrays:
                description_arrays[data_key] = (
                    value.long_description
                    if value.long_description
                    else value.short_description
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
            catdesc=description_arrays[key],
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
