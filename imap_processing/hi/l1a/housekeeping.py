"""Unpack IMAP-Hi housekeeping data."""
import collections
import dataclasses

import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.hi.hi_cdf_attrs import hi_hk_l1a_attrs


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
            metadata_arrays.setdefault(key, []).append(value.raw_value)
            # add it once since description should be same for all packets
            if key not in description_arrays:
                description_arrays[key] = (
                    value.long_description
                    if value.long_description
                    else value.short_description
                )
        for key, value in data_packet.data.items():
            metadata_arrays.setdefault(key, []).append(value.raw_value)
            # add it once since description should be same for all packets
            if key not in description_arrays:
                description_arrays[key] = (
                    value.long_description
                    if value.long_description
                    else value.short_description
                )

    epoch_time = xr.DataArray(
        metadata_arrays["CCSDS_MET"],
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch_time},
    )

    # create xarray dataset for each metadata field
    for key, value in metadata_arrays.items():
        # replace description and fieldname
        data_attrs = dataclasses.replace(
            hi_hk_l1a_attrs,
            catdesc=description_arrays[key],
            fieldname=key,
        )
        dataset[key] = xr.DataArray(
            value,
            dims=["epoch"],
            attrs=data_attrs.output(),
        )
    return dataset
