"""
Current File contains functions that every instrument can use.

1. sort_by_time function
    Sorts data packets list by time. It takes packets list
    and time key as input. Then sorts packets by time key input
2. group_by_apid function
    Group packets by apid. It takes packets list as input.
3. convert_raw_to_eu function
    Convert raw data to engineering unit. It read conversion
    table from the given input path and reads current
    packet name's metadata fields. Then it converts raw data
    to engineering unit.
"""
import collections

import numpy as np
import pandas as pd
import xarray as xr


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
        # expecting coefficient in decending order
        coeff_values = metadata_list.loc[
            metadata_list["mnemonic"] == metadata_field
        ].values[0][6:][::-1]

        # Convert the raw value to engineering unit
        dataset[metadata_field].data = np.polyval(
            coeff_values, dataset[metadata_field].data
        )

    return dataset
