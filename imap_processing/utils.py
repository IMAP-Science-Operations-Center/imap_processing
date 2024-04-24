"""Common functions that every instrument can use."""

import collections
import dataclasses

import numpy as np
import pandas as pd
import xarray as xr
from space_packet_parser.parser import Packet

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.common_cdf_attrs import metadata_attrs


def sort_by_time(packets, time_key):
    """Sort packets by specified key.

    Parameters
    ----------
    packets : list
        Decom data packets
    time_key : str
        Key to sort by. Must be a key in the packets data dictionary.
        e.g. "SHCOARSE" or "MET_TIME" or "ACQ_START_COARSE"

    Returns
    -------
    list
        sorted packets
    """
    sorted_packets = sorted(packets, key=lambda x: x.data[time_key].raw_value)
    return sorted_packets


def group_by_apid(packets: list):
    """Group data by apid.

    Parameters
    ----------
    packets : list
        packet list

    Returns
    -------
    dict
        grouped data by apid
    """
    grouped_packets = collections.defaultdict(list)
    for packet in packets:
        apid = packet.header["PKT_APID"].raw_value
        grouped_packets.setdefault(apid, []).append(packet)
    return grouped_packets


def convert_raw_to_eu(dataset: xr.Dataset, conversion_table_path, packet_name):
    """Convert raw data to engineering unit.

    Parameters
    ----------
    dataset : xr.Dataset
        Raw data.
    conversion_table_path : str
        Path to engineering unit conversion table.
        Eg:
        f"{imap_module_directory}/swe/l1b/engineering_unit_convert_table.csv"
    packet_name: str
        Packet name

    Returns
    -------
    xr.Dataset
        Raw data converted to engineering unit as needed.
    """
    # Make sure there is column called "index" with unique
    # value such as 0, 1, 2, 3, ...
    eu_conversion_table = pd.read_csv(
        conversion_table_path,
        index_col="index",
    )

    # Look up all metadata fields for the packet name
    metadata_list = eu_conversion_table.loc[
        eu_conversion_table["packetName"] == packet_name
    ]

    # for each metadata field, convert raw value to engineering unit
    for field in metadata_list.index:
        metadata_field = metadata_list.loc[field]["mnemonic"]
        # On this line, we are getting the coefficients from the
        # table and then reverse them because the np.polyval is
        # expecting coefficient in descending order
        coeff_values = metadata_list.loc[
            metadata_list["mnemonic"] == metadata_field
        ].values[0][6:][::-1]

        # Convert the raw value to engineering unit
        dataset[metadata_field].data = np.polyval(
            coeff_values, dataset[metadata_field].data
        )

    return dataset


def create_dataset(
    packets: list[Packet],
    spacecraft_time_key="shcoarse",
    include_header=True,
    skip_keys=None,
) -> xr.Dataset:
    """Create dataset for each metadata field.

    Parameters
    ----------
    packets : list[Packet]
        packet list
    spacecraft_time_key : str, Optional
        Default is "shcoarse" because many instrument uses it.
        This key is used to get spacecraft time for epoch dimension.
    include_header: bool, Optional
        Whether to include CCSDS header data in the dataset
    skip_keys: list, Optional
        Keys to skip in the metadata

    Returns
    -------
    xr.dataset
        dataset with all metadata field data in xr.DataArray
    """
    metadata_arrays = collections.defaultdict(list)
    description_dict = {}

    for data_packet in packets:
        data_to_include = (
            (data_packet.header | data_packet.data)
            if include_header
            else data_packet.data
        )

        # Drop keys using skip_keys
        if skip_keys is not None:
            for key in skip_keys:
                data_to_include.pop(key, None)

        # Add metadata to array
        for key, value in data_to_include.items():
            # convert key to lower case to match SPDF requirement
            data_key = key.lower()
            metadata_arrays[data_key].append(value.raw_value)
            # description should be same for all packets
            description_dict[data_key] = (
                value.long_description or value.short_description
            )

    epoch_time = xr.DataArray(
        metadata_arrays[spacecraft_time_key],
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
            metadata_attrs,
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
