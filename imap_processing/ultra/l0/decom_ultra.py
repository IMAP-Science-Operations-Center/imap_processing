"""Decommutates Ultra CCSDS packets."""

import collections
import logging
from collections import defaultdict
from typing import Any, Union

import numpy as np
import xarray as xr
from space_packet_parser import packets

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.ultra.l0.decom_tools import (
    decompress_binary,
    decompress_image,
    read_image_raw_events_binary,
)
from imap_processing.ultra.l0.ultra_utils import (
    EVENT_FIELD_RANGES,
    RATES_KEYS,
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
    append_ccsds_fields,
)
from imap_processing.utils import convert_to_binary_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def append_tof_params(
    decom_data: dict,
    packet: packets.CCSDSPacket,
    decompressed_data: np.ndarray,
    data_dict: dict,
    stacked_dict: dict,
) -> None:
    """
    Append parsed items to a dictionary, including decompressed data if available.

    Parameters
    ----------
    decom_data : dict
        Dictionary to which the data is appended.
    packet : space_packet_parser.packets.CCSDSPacket
        Individual packet.
    decompressed_data : list
        Data that has been decompressed.
    data_dict : dict
        Dictionary used for stacking in SID dimension.
    stacked_dict : dict
        Dictionary used for stacking in time dimension.
    """
    # TODO: add error handling to make certain every timestamp has 8 SID values

    for key in packet.user_data.keys():
        # Keep appending packet data until SID = 7
        if key == "PACKETDATA":
            data_dict[key].append(decompressed_data)
        # Keep appending all other data until SID = 7
        else:
            data_dict[key].append(packet[key])

    # Append CCSDS fields to the dictionary
    ccsds_data = CcsdsData(packet.header)
    append_ccsds_fields(data_dict, ccsds_data)

    # Once "SID" reaches 7, we have all the images and data for the single timestamp
    if packet["SID"] == 7:
        decom_data["SHCOARSE"].extend(list(set(data_dict["SHCOARSE"])))
        data_dict["SHCOARSE"].clear()

        for key in packet.user_data.keys():
            if key != "SHCOARSE":
                stacked_dict[key].append(np.stack(data_dict[key]))
                data_dict[key].clear()
        for key in packet.header.keys():
            stacked_dict[key].append(np.stack(data_dict[key]))
            data_dict[key].clear()


def append_params(decom_data: dict, packet: packets.CCSDSPacket) -> None:
    # Todo Update what packet type is.
    """
    Append parsed items to a dictionary, including decompressed data if available.

    Parameters
    ----------
    decom_data : dict
        Dictionary to which the data is appended.
    packet : space_packet_parser.packets.CCSDSPacket
        Individual packet.
    """
    for key, value in packet.user_data.items():
        decom_data[key].append(value)

    ccsds_data = CcsdsData(packet.header)
    append_ccsds_fields(decom_data, ccsds_data)


def process_ultra_apids(data: list, apid: int) -> Union[dict[Any, Any], bool]:
    """
    Unpack and decode Ultra packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    data : list
        Grouped data.
    apid : int
        The APID to process.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    # Strategy dict maps APIDs to their respective processing functions
    strategy_dict = {
        ULTRA_TOF.apid[0]: process_ultra_tof,
        ULTRA_EVENTS.apid[0]: process_ultra_events,
        ULTRA_AUX.apid[0]: process_ultra_aux,
        ULTRA_RATES.apid[0]: process_ultra_rates,
    }

    process_function = strategy_dict.get(apid, lambda *args: False)
    decom_data = process_function(data, defaultdict(list))

    return decom_data


def process_ultra_tof(ds: xr.Dataset, decom_data: collections.defaultdict) -> dict:
    """
    Unpack and decode Ultra TOF packets.

    Parameters
    ----------
    ds : xarray.Dataset
        TOF dataset.
    decom_data : collections.defaultdict
        Empty dictionary.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    stacked_dict: dict = defaultdict(list)
    data_dict: dict = defaultdict(list)

    # For TOF we need to sort by time and then SID
    ds = ds.sortby(["epoch", "sid"])

    if isinstance(ULTRA_TOF.mantissa_bit_length, int) and isinstance(
        ULTRA_TOF.width, int
    ):
        for epoch in ds["epoch"]:
            packet = ds.sel(epoch=epoch)
            binary_data = convert_to_binary_string(packet["PACKETDATA"])
            # Decompress the image data
            decompressed_data = decompress_image(
                packet["P00"],
                binary_data,
                ULTRA_TOF.width,
                ULTRA_TOF.mantissa_bit_length,
            )

            # Append the decompressed data and other derived data
            # to the dictionary
            append_tof_params(
                decom_data,
                packet,
                decompressed_data=decompressed_data,
                data_dict=data_dict,
                stacked_dict=stacked_dict,
            )

    # Stack the data to create required dimensions
    for key, value in stacked_dict.items():
        decom_data[key] = np.stack(value)

    return decom_data


def process_ultra_events(sorted_packets: xr.Dataset, decom_data: dict) -> xr.Dataset:
    """
    Unpack and decode Ultra EVENTS packets.

    Parameters
    ----------
    sorted_packets : xr.Dataset
        EVENTS packets sorted by time.
    decom_data : collections.defaultdict
        Empty dictionary.

    Returns
    -------
    event_dataset : xr.Dataset
        EVENTS packets containing the decoded data.
    """
    all_events = []
    all_indices = []
    EMPTY_EVENT = {field: np.iinfo(np.int64).min for field in EVENT_FIELD_RANGES}
    counts = sorted_packets["count"].values
    eventdata_array = sorted_packets["eventdata"].values

    for i, count in enumerate(counts):
        if count == 0:
            all_events.append(EMPTY_EVENT)
            all_indices.append(i)
        else:
            # Here there are multiple images in a single packet,
            # so we need to loop through each image and decompress it.
            event_data_list = read_image_raw_events_binary(
                eventdata_array[i], count, decom_data
            )
            all_events.extend(event_data_list)
            all_indices.extend([i] * count)

    event_fields = all_events[0].keys()
    event_data = {
        field: np.array([ev[field] for ev in all_events]) for field in event_fields
    }

    idx = np.array(all_indices)

    metadata = {
        var: (["event"], sorted_packets[var].values[idx])
        for var in sorted_packets.data_vars
        if var != "eventdata"
    }

    coords = {
        coord: (["event"], sorted_packets[coord].values[idx])
        for coord in sorted_packets.coords
    }

    event_dataset = xr.Dataset(
        data_vars={**metadata, **{k: ("event", v) for k, v in event_data.items()}},
        coords={"event": np.arange(len(idx)), **coords},
    )

    return event_dataset


def process_ultra_aux(sorted_packets: list, decom_data: dict) -> dict:
    """
    Unpack and decode Ultra AUX packets.

    Parameters
    ----------
    sorted_packets : list
        AUX packets sorted by time.
    decom_data : collections.defaultdict
        Empty dictionary.

    Returns
    -------
    decom_data : dict
        A dictionary containing the decoded data.
    """
    for packet in sorted_packets:
        append_params(decom_data, packet)

    return decom_data


def process_ultra_rates(sorted_packets: xr.Dataset, decom_data: dict) -> xr.Dataset:
    """
    Unpack and decode Ultra RATES packets.

    Parameters
    ----------
    sorted_packets : xr.Dataset
        RATES packets sorted by time.
    decom_data : collections.defaultdict
        Empty dictionary.

    Returns
    -------
    sorted_packets : xr.Dataset
        RATES packets containing the decoded data.
    """
    if (
        isinstance(ULTRA_RATES.mantissa_bit_length, int)
        and isinstance(ULTRA_RATES.len_array, int)
        and isinstance(ULTRA_RATES.block, int)
        and isinstance(ULTRA_RATES.width, int)
    ):
        for fastdata in sorted_packets["fastdata_00"]:
            raw_binary_string = convert_to_binary_string(fastdata.item())
            decompressed_data = decompress_binary(
                raw_binary_string,
                ULTRA_RATES.width,
                ULTRA_RATES.block,
                ULTRA_RATES.len_array,
                ULTRA_RATES.mantissa_bit_length,
            )

            for index in range(ULTRA_RATES.len_array):
                decom_data[RATES_KEYS[index].lower()].append(decompressed_data[index])

        for key, values in decom_data.items():
            sorted_packets[key] = xr.DataArray(np.array(values), dims=["epoch"])

    return sorted_packets
