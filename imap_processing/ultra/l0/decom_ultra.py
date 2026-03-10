"""Decommutates Ultra CCSDS packets."""

import logging
import math
from collections import defaultdict
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ultra.l0.decom_tools import (
    decompress_binary,
    decompress_image,
    read_image_raw_events_binary,
)
from imap_processing.ultra.l0.ultra_utils import (
    CMD_ECHO_MAP,
    ENERGY_EVENT_FIELD_RANGES,
    ENERGY_RATES_KEYS,
    EVENT_FIELD_RANGES,
    RATES_KEYS,
    ULTRA_ENERGY_EVENTS,
    ULTRA_ENERGY_RATES,
    ULTRA_ENERGY_SPECTRA,
    ULTRA_EVENTS,
    ULTRA_PRI_1_EVENTS,
    ULTRA_PRI_2_EVENTS,
    ULTRA_PRI_3_EVENTS,
    ULTRA_PRI_4_EVENTS,
    ULTRA_RATES,
    PacketProperties,
)
from imap_processing.utils import combine_segmented_packets, convert_to_binary_string

logger = logging.getLogger(__name__)


def extract_initial_items_from_combined_packets(
    packets: xr.Dataset,
) -> xr.Dataset:
    """
    Extract metadata fields from the beginning of combined event_data packets.

    Extracts bit fields from the first 20 bytes of each event_data array
    and adds them as new variables to the dataset.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset containing combined packets with event_data.

    Returns
    -------
    xarray.Dataset
        Dataset with extracted metadata fields added.
    """
    # Initialize arrays for extracted fields
    n_packets = len(packets.epoch)

    # Preallocate arrays
    sid = np.zeros(n_packets, dtype=np.uint8)
    spin = np.zeros(n_packets, dtype=np.uint8)
    abortflag = np.zeros(n_packets, dtype=np.uint8)
    startdelay = np.zeros(n_packets, dtype=np.uint16)
    p00 = np.zeros(n_packets, dtype=np.uint8)

    # Extract the data array outside of the loop
    binary_data = packets["packetdata"].data
    # Extract fields from each packet
    for pkt_idx in range(n_packets):
        event_data = binary_data[pkt_idx]

        sid[pkt_idx] = event_data[0]
        spin[pkt_idx] = event_data[1]
        abortflag[pkt_idx] = (event_data[2] >> 7) & 0x1
        startdelay[pkt_idx] = int.from_bytes(event_data[2:4], byteorder="big") & 0x7FFF
        p00[pkt_idx] = event_data[4]

        # Remove the first 5 bytes after extraction
        binary_data[pkt_idx] = event_data[5:]

    # Add extracted fields to dataset
    packets["sid"] = xr.DataArray(sid, dims=["epoch"])
    packets["spin"] = xr.DataArray(spin, dims=["epoch"])
    packets["abortflag"] = xr.DataArray(abortflag, dims=["epoch"])
    packets["startdelay"] = xr.DataArray(startdelay, dims=["epoch"])
    packets["p00"] = xr.DataArray(p00, dims=["epoch"])

    return packets


def process_ultra_tof(ds: xr.Dataset, packet_props: PacketProperties) -> xr.Dataset:
    """
    Unpack and decode Ultra TOF packets.

    The TOF packets contain image data that may be split across multiple segmented
    packets. This function combines the segmented packets and decompresses the image
    data.

    Parameters
    ----------
    ds : xarray.Dataset
        TOF dataset.
    packet_props : PacketProperties
        Information that defines properties of the packet including the pixel window
        dimensions of images and number of image panes.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the decoded and decompressed data.
    """
    # Combine segmented packets
    ds = combine_segmented_packets(ds, binary_field_name="packetdata")
    # Extract the header keys from each of the combined packetdata fields.
    ds = extract_initial_items_from_combined_packets(ds)

    scalar_keys = [key for key in ds.data_vars if key not in ("packetdata", "sid")]

    image_planes = packet_props.image_planes
    rows = packet_props.pixel_window_rows
    cols = packet_props.pixel_window_columns
    planes_per_packet = packet_props.image_planes_per_packet

    if (
        image_planes is None
        or rows is None
        or cols is None
        or planes_per_packet is None
    ):
        raise ValueError(
            "Packet properties must specify pixel window dimensions, "
            "width bit, image planes, and image planes per packet for this packet type."
        )
    # Calculate the number of image packets based on the number of image panes and
    # planes per packet.
    # There may be cases where the last packet has fewer planes than the
    # planes_per_packet, to account for this, we use ceiling division.
    num_image_packets = math.ceil(image_planes / planes_per_packet)

    decom_data: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    decom_data["packetdata"] = []
    valid_epoch = []
    for val, group in ds.groupby("epoch"):
        if set(group["sid"].values) >= set(
            np.arange(0, image_planes, planes_per_packet)
        ):
            plane_count = 0
            valid_epoch.append(val)
            group.sortby("sid")

            for key in scalar_keys:
                # Repeat the scalar values for each image plane. There may be cases
                # where the last packet has fewer planes than the planes_per_packet, so
                # we slice to ensure the correct length.
                decom_data[key].append(
                    np.tile(group[key].values, planes_per_packet)[:image_planes]
                )

            image = []
            for i in range(num_image_packets):
                binary = convert_to_binary_string(group["packetdata"].values[i])
                # Determine how many planes to decompress in this packet.
                # the last packet might have fewer planes than planes_per_packet.
                # Take the minimum of the remaining planes or the max planes per packet
                # value.
                planes_in_packet = min(image_planes - plane_count, planes_per_packet)
                decompressed = decompress_image(
                    group["p00"].values[i],
                    binary,
                    packet_props,
                    planes_in_packet,
                )
                image.append(decompressed)
                plane_count += planes_in_packet

            decom_data["packetdata"].append(np.concatenate(image, axis=0))

    for key in scalar_keys:
        decom_data[key] = np.stack(decom_data[key], axis=0)

    decom_data["packetdata"] = np.stack(decom_data["packetdata"], axis=0)

    coords = {
        "epoch": np.array(valid_epoch, dtype=np.uint64),
        "plane": xr.DataArray(np.arange(image_planes), dims=["plane"], name="plane"),
        "row": xr.DataArray(np.arange(rows), dims=["row"], name="row"),
        "column": xr.DataArray(np.arange(cols), dims=["column"], name="column"),
    }

    dataset = xr.Dataset(coords=coords)

    # Add scalar keys (2D: epoch x packets)
    for key in scalar_keys:
        dataset[key] = xr.DataArray(
            decom_data[key],
            dims=["epoch", "plane"],
        )

    # Add PACKETDATA (4D: epoch x sid x row x column)
    dataset["packetdata"] = xr.DataArray(
        decom_data["packetdata"],
        dims=["epoch", "plane", "row", "column"],
    )

    return dataset


def get_event_id(shcoarse: NDArray) -> NDArray:
    """
    Get unique event IDs using data from events packets.

    Parameters
    ----------
    shcoarse : numpy.ndarray
        SHCOARSE (MET).

    Returns
    -------
    event_ids : numpy.ndarray
        Ultra events data with calculated unique event IDs as 64-bit integers.
    """
    event_ids = []
    packet_counters = {}

    for met in shcoarse:
        # Initialize the counter for a new packet (MET value)
        if met not in packet_counters:
            packet_counters[met] = 0
        else:
            packet_counters[met] += 1

        # Left shift SHCOARSE (u32) by 31 bits, to make room for our event counters
        # (31 rather than 32 to keep it positive in the int64 representation)
        # Append the current number of events in this packet to the right-most bits
        # This makes each event a unique value including the MET and event number
        # in the packet
        # NOTE: CDF does not allow for uint64 values,
        # so we use int64 representation here
        event_id = (np.int64(met) << np.int64(31)) | np.int64(packet_counters[met])
        event_ids.append(event_id)

    return np.array(event_ids, dtype=np.int64)


def process_ultra_events(ds: xr.Dataset, apid: int) -> xr.Dataset:
    """
    Unpack and decode Ultra EVENTS packets.

    Parameters
    ----------
    ds : xarray.Dataset
        Events dataset.
    apid : int
        APID of the events dataset.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the decoded and decompressed data.
    """
    all_event_apids = set(
        ULTRA_EVENTS.apid
        + ULTRA_PRI_1_EVENTS.apid
        + ULTRA_PRI_2_EVENTS.apid
        + ULTRA_PRI_3_EVENTS.apid
        + ULTRA_PRI_4_EVENTS.apid
    )
    if apid in all_event_apids:
        field_ranges = EVENT_FIELD_RANGES
    elif apid in ULTRA_ENERGY_EVENTS.apid:
        field_ranges = ENERGY_EVENT_FIELD_RANGES
    else:
        raise ValueError(f"APID {apid} not recognized for Ultra events processing.")

    all_events = []
    all_indices = []

    attrs = ImapCdfAttributes()
    attrs.add_instrument_variable_attrs("ultra", level="l1a")

    empty_event = {
        field: attrs.get_variable_attributes(field).get(
            "FILLVAL", np.iinfo(np.int64).min
        )
        for field in field_ranges
    }

    counts = ds["count"].values
    eventdata_array = ds["eventdata"].values

    for i, count in enumerate(counts):
        if count == 0:
            all_events.append(empty_event)
            all_indices.append(i)
        else:
            # Here there are multiple images in a single packet,
            # so we need to loop through each image and decompress it.
            event_data_list = read_image_raw_events_binary(
                eventdata_array[i], count, field_ranges
            )
            all_events.extend(event_data_list)
            # Keep track of how many times does the event occurred at this epoch.
            all_indices.extend([i] * count)

    # Now we have the event data, we need to create the xarray dataset.
    # We cannot append to the existing dataset (sorted_packets)
    # because there are multiple events for each epoch.
    idx = np.array(all_indices)

    # Expand the existing dataset so that it is the same length as the event data.
    expanded_data = {
        var: ds[var].values[idx] for var in ds.data_vars if var != "eventdata"
    }

    # Add the event data to the expanded dataset.
    for key in field_ranges:
        expanded_data[key] = np.array([event[key] for event in all_events])

    event_ids = get_event_id(expanded_data["shcoarse"])

    coords = {
        "epoch": ds["epoch"].values[idx],
        "event_id": ("epoch", event_ids),
    }

    dataset = xr.Dataset(coords=coords)
    for key, data in expanded_data.items():
        dataset[key] = xr.DataArray(
            data,
            dims=["epoch"],
        )

    return dataset


def process_ultra_rates(ds: xr.Dataset) -> xr.Dataset:
    """
    Unpack and decode Ultra RATES packets.

    Parameters
    ----------
    ds : xarray.Dataset
       Rates dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the decoded and decompressed data.
    """
    decom_data = defaultdict(list)

    for fastdata in ds["fastdata_00"]:
        raw_binary_string = convert_to_binary_string(fastdata.item())
        decompressed_data = decompress_binary(
            raw_binary_string,
            cast(int, ULTRA_RATES.width),
            cast(int, ULTRA_RATES.block),
            cast(int, ULTRA_RATES.len_array),
            cast(int, ULTRA_RATES.mantissa_bit_length),
        )

        for index in range(cast(int, ULTRA_RATES.len_array)):
            decom_data[RATES_KEYS[index]].append(decompressed_data[index])

    for key, values in decom_data.items():
        ds[key] = xr.DataArray(np.array(values), dims=["epoch"])

    return ds


def process_ultra_energy_rates(ds: xr.Dataset) -> xr.Dataset:
    """
    Unpack and decode Ultra ENERGY RATES packets.

    Parameters
    ----------
    ds : xarray.Dataset
       Energy rates dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the decoded and decompressed data.
    """
    decom_data = defaultdict(list)

    for rate in ds["ratedata"]:
        raw_binary_string = convert_to_binary_string(rate.item())
        decompressed_data = decompress_binary(
            raw_binary_string,
            cast(int, ULTRA_ENERGY_RATES.width),
            cast(int, ULTRA_ENERGY_RATES.block),
            cast(int, ULTRA_ENERGY_RATES.len_array),
            cast(int, ULTRA_ENERGY_RATES.mantissa_bit_length),
        )

        for index in range(cast(int, ULTRA_ENERGY_RATES.len_array)):
            decom_data[ENERGY_RATES_KEYS[index]].append(decompressed_data[index])

    for key, values in decom_data.items():
        ds[key] = xr.DataArray(np.array(values), dims=["epoch"])

    return ds


def process_ultra_energy_spectra(ds: xr.Dataset) -> xr.Dataset:
    """
    Unpack and decode Ultra ENERGY SPECTRA packets.

    Parameters
    ----------
    ds : xarray.Dataset
       Energy rates dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the decoded and decompressed data.
    """
    energy_spectra = []

    for rate in ds["compdata"]:
        raw_binary_string = convert_to_binary_string(rate.item())
        decompressed_data = decompress_binary(
            raw_binary_string,
            cast(int, ULTRA_ENERGY_SPECTRA.width),
            cast(int, ULTRA_ENERGY_SPECTRA.block),
            cast(int, ULTRA_ENERGY_SPECTRA.len_array),
            cast(int, ULTRA_ENERGY_SPECTRA.mantissa_bit_length),
        )

        energy_spectra.append(decompressed_data)

    energy_spectra = np.array(energy_spectra)

    ds["ssd_sum"] = xr.DataArray(
        energy_spectra,
        dims=["epoch", "energyspectrastate"],
        coords={"epoch": ds["epoch"], "energyspectrastate": np.arange(16)},
    )

    return ds


def process_ultra_cmd_echo(ds: xr.Dataset) -> xr.Dataset:
    """
    Unpack and decode Ultra CMD ECHO packets.

    Parameters
    ----------
    ds : xarray.Dataset
       Energy rates dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the decoded and decompressed data.
    """
    descriptions = []

    fill = 0xFF
    max_len = 10
    arg_array = np.full((len(ds["epoch"]), max_len), fill, dtype=np.uint8)

    for i, arg in enumerate(ds["args"].values):
        # Converts to the numeric representations of each byte.
        arg_array[i, : len(arg)] = np.frombuffer(arg, dtype=np.uint8)

    # Default to "FILL" for unlisted values
    for result in ds["result"].values:
        descriptions.append(CMD_ECHO_MAP.get(result, "FILL"))

    ds["arguments"] = xr.DataArray(
        arg_array,
        dims=["epoch", "arg_index"],
        coords={
            "epoch": ds["epoch"],
            "arg_index": np.arange(10),
        },
    )

    ds["result_description"] = xr.DataArray(
        np.array(descriptions),
        dims=["epoch"],
        coords={"epoch": ds["epoch"]},
    )

    ds = ds.drop_vars(["args", "result"])

    return ds


def process_ultra_macros_checksum(ds: xr.Dataset) -> xr.Dataset:
    """
    Unpack and decode Ultra MACROS CHECKSUM packets.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing macro checksums.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with unpacked and decoded checksum values.
    """
    # big endian uint16
    packed_dtype = np.dtype(">u2")
    fill = np.iinfo(packed_dtype).max
    n_epochs = ds.sizes["epoch"]
    max_len = 256

    checksum_array = np.full((n_epochs, max_len), fill)

    for i, checksum in enumerate(ds["checksums"]):
        checksum_array[i, :] = np.frombuffer(checksum.item(), dtype=packed_dtype)

    ds["checksum"] = xr.DataArray(
        checksum_array,
        dims=["epoch", "checksum_index"],
        coords={"epoch": ds["epoch"], "checksum_index": np.arange(max_len)},
    )
    ds = ds.drop_vars(["checksums"])

    return ds
