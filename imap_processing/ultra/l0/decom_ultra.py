"""Decommutates Ultra CCSDS packets."""

import logging
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
    ULTRA_TOF,
)
from imap_processing.utils import convert_to_binary_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_ultra_tof(ds: xr.Dataset) -> xr.Dataset:
    """
    Unpack and decode Ultra TOF packets.

    Parameters
    ----------
    ds : xarray.Dataset
        TOF dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the decoded and decompressed data.
    """
    scalar_keys = [key for key in ds.data_vars if key not in ("packetdata", "sid")]

    decom_data: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    decom_data["packetdata"] = []
    valid_epoch = []
    width = cast(int, ULTRA_TOF.width)
    mantissa_bit_length = cast(int, ULTRA_TOF.mantissa_bit_length)

    for val, group in ds.groupby("epoch"):
        if set(group["sid"].values) >= set(range(8)):
            valid_epoch.append(val)
            group.sortby("sid")

            for key in scalar_keys:
                decom_data[key].append(group[key].values)

            image = []
            for i in range(8):
                binary = convert_to_binary_string(group["packetdata"].values[i])
                decompressed = decompress_image(
                    group["p00"].values[i],
                    binary,
                    width,
                    mantissa_bit_length,
                )
                image.append(decompressed)

            decom_data["packetdata"].append(np.stack(image))

    for key in scalar_keys:
        decom_data[key] = np.stack(decom_data[key])

    decom_data["packetdata"] = np.stack(decom_data["packetdata"])

    coords = {
        "epoch": np.array(valid_epoch, dtype=np.uint64),
        "sid": xr.DataArray(np.arange(8), dims=["sid"], name="sid"),
        "row": xr.DataArray(np.arange(54), dims=["row"], name="row"),
        "column": xr.DataArray(np.arange(180), dims=["column"], name="column"),
    }

    dataset = xr.Dataset(coords=coords)

    # Add scalar keys (2D: epoch x sid)
    for key in scalar_keys:
        dataset[key] = xr.DataArray(
            decom_data[key],
            dims=["epoch", "sid"],
        )

    # Add PACKETDATA (4D: epoch x sid x row x column)
    dataset["packetdata"] = xr.DataArray(
        decom_data["packetdata"],
        dims=["epoch", "sid", "row", "column"],
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
