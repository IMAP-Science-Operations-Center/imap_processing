"""Processing functions for CoDICE L1A Direct Event data."""

import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice import constants
from imap_processing.codice.decompress import decompress
from imap_processing.codice.utils import (
    CODICEAPID,
    CoDICECompression,
    ViewTabInfo,
    apply_replacements_to_attrs,
    get_codice_epoch_time,
)
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.utils import combine_segmented_packets

logger = logging.getLogger(__name__)


def extract_initial_items_from_combined_packets(
    packets: xr.Dataset,
) -> xr.Dataset:
    """
    Extract fields from the beginning of combined event_data packets.

    Extracts bit fields from the first 20 bytes of each event_data array
    and add them as new variables to the dataset.

    This was previously done in XTCE, but we can't do that because of
    segmented packets that need to be combined. Each segmented packet
    has its own (SHCOARSE, EVENTDATA, CHKSUM) fields, so we need to
    only combine along the EVENTDATA field and extract data that way.

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
    packet_version = np.zeros(n_packets, dtype=np.uint16)
    spin_period = np.zeros(n_packets, dtype=np.uint16)
    acq_start_seconds = np.zeros(n_packets, dtype=np.uint32)
    acq_start_subseconds = np.zeros(n_packets, dtype=np.uint32)
    spare_1 = np.zeros(n_packets, dtype=np.uint8)
    st_bias_gain_mode = np.zeros(n_packets, dtype=np.uint8)
    sw_bias_gain_mode = np.zeros(n_packets, dtype=np.uint8)
    suspect = np.zeros(n_packets, dtype=np.uint8)
    priority = np.zeros(n_packets, dtype=np.uint8)
    compressed = np.zeros(n_packets, dtype=np.uint8)
    rgfo_half_spin = np.zeros(n_packets, dtype=np.uint8)
    rgfo_esa_step = np.zeros(n_packets, dtype=np.uint8)
    rgfo_spin_sector = np.zeros(n_packets, dtype=np.uint8)
    nso_half_spin = np.zeros(n_packets, dtype=np.uint8)
    nso_spin_sector = np.zeros(n_packets, dtype=np.uint8)
    nso_esa_step = np.zeros(n_packets, dtype=np.uint8)
    spare_2 = np.zeros(n_packets, dtype=np.uint16)
    num_events = np.zeros(n_packets, dtype=np.uint32)
    byte_count = np.zeros(n_packets, dtype=np.uint32)

    # Extract fields from each packet
    for pkt_idx in range(n_packets):
        event_data = packets.event_data.data[pkt_idx]

        # Byte-aligned fields using int.from_bytes
        packet_version[pkt_idx] = int.from_bytes(event_data[0:2], byteorder="big")
        spin_period[pkt_idx] = int.from_bytes(event_data[2:4], byteorder="big")
        acq_start_seconds[pkt_idx] = int.from_bytes(event_data[4:8], byteorder="big")

        # Non-byte-aligned fields (bytes 8-12 contain mixed bit fields)
        # Extract 4 bytes and unpack bit fields
        mixed_bytes = int.from_bytes(event_data[8:12], byteorder="big")

        # acq_start_subseconds: 20 bits (MSB)
        acq_start_subseconds[pkt_idx] = (mixed_bytes >> 12) & 0xFFFFF
        # spare_1: 2 bits
        spare_1[pkt_idx] = (mixed_bytes >> 10) & 0x3
        # st_bias_gain_mode: 2 bits
        st_bias_gain_mode[pkt_idx] = (mixed_bytes >> 8) & 0x3
        # sw_bias_gain_mode: 2 bits
        sw_bias_gain_mode[pkt_idx] = (mixed_bytes >> 6) & 0x3
        # priority: 4 bits
        priority[pkt_idx] = (mixed_bytes >> 2) & 0xF
        # suspect: 1 bit
        suspect[pkt_idx] = (mixed_bytes >> 1) & 0x1
        # compressed: 1 bit (LSB)
        compressed[pkt_idx] = mixed_bytes & 0x1
        # After packet version 1, the fields below are present in event_data
        if packet_version[pkt_idx] > 1:
            # All of the fields below are single byte fields
            rgfo_half_spin[pkt_idx] = event_data[12]
            rgfo_spin_sector[pkt_idx] = event_data[13]
            rgfo_esa_step[pkt_idx] = event_data[14]
            nso_half_spin[pkt_idx] = event_data[15]
            nso_spin_sector[pkt_idx] = event_data[16]
            nso_esa_step[pkt_idx] = event_data[17]

            # spare_2 is 16 bits
            spare_2[pkt_idx] = int.from_bytes(event_data[18:20], byteorder="big")
            # Remaining byte-aligned fields
            num_events[pkt_idx] = int.from_bytes(event_data[20:24], byteorder="big")
            byte_count[pkt_idx] = int.from_bytes(event_data[24:28], byteorder="big")
            # Header is 28 bytes total for version > 1
            len_header = 28
        else:
            # Remaining byte-aligned fields
            num_events[pkt_idx] = int.from_bytes(event_data[12:16], byteorder="big")
            byte_count[pkt_idx] = int.from_bytes(event_data[16:20], byteorder="big")
            # Header is 20 bytes total for version 1
            len_header = 20

        # Remove the first len_header bytes from event_data (header fields from above)
        # Then trim to the number of bytes indicated by byte_count
        if byte_count[pkt_idx] > len(event_data) - len_header:
            raise ValueError(
                f"Byte count {byte_count[pkt_idx]} exceeds available "
                f"data length {len(event_data) - len_header} for packet index"
                f" {pkt_idx}."
            )

        packets.event_data.data[pkt_idx] = event_data[
            len_header : byte_count[pkt_idx] + len_header
        ]
        if compressed[pkt_idx]:
            packets.event_data.data[pkt_idx] = decompress(
                packets.event_data.data[pkt_idx],
                CoDICECompression.LOSSLESS,
            )

    # Add extracted fields to dataset
    packets["packet_version"] = xr.DataArray(packet_version, dims=["epoch"])
    packets["spin_period"] = xr.DataArray(spin_period, dims=["epoch"])
    packets["acq_start_seconds"] = xr.DataArray(acq_start_seconds, dims=["epoch"])
    packets["acq_start_subseconds"] = xr.DataArray(acq_start_subseconds, dims=["epoch"])
    packets["spare_1"] = xr.DataArray(spare_1, dims=["epoch"])
    packets["st_bias_gain_mode"] = xr.DataArray(st_bias_gain_mode, dims=["epoch"])
    packets["sw_bias_gain_mode"] = xr.DataArray(sw_bias_gain_mode, dims=["epoch"])
    packets["priority"] = xr.DataArray(priority, dims=["epoch"])
    packets["suspect"] = xr.DataArray(suspect, dims=["epoch"])
    packets["compressed"] = xr.DataArray(compressed, dims=["epoch"])
    packets["rgfo_half_spin"] = xr.DataArray(rgfo_half_spin, dims=["epoch"])
    packets["rgfo_spin_sector"] = xr.DataArray(rgfo_spin_sector, dims=["epoch"])
    packets["rgfo_esa_step"] = xr.DataArray(rgfo_esa_step, dims=["epoch"])
    packets["nso_half_spin"] = xr.DataArray(nso_half_spin, dims=["epoch"])
    packets["nso_spin_sector"] = xr.DataArray(nso_spin_sector, dims=["epoch"])
    packets["nso_esa_step"] = xr.DataArray(nso_esa_step, dims=["epoch"])
    packets["spare_2"] = xr.DataArray(spare_2, dims=["epoch"])
    packets["num_events"] = xr.DataArray(num_events, dims=["epoch"])
    packets["byte_count"] = xr.DataArray(byte_count, dims=["epoch"])

    return packets


def unpack_bits(bit_structure: dict, de_data: np.ndarray) -> dict:
    """
    Unpack 64-bit values into separate fields based on bit structure.

    Parameters
    ----------
    bit_structure : dict
        Dictionary mapping variable names to their bit lengths.
    de_data : np.ndarray
        1D array of 64-bit values to unpack.

    Returns
    -------
    dict
        Dictionary of field_name -> unpacked values array.
    """
    unpacked = {}
    # Data need to be unpacked in right to left order (LSB). Eg.
    #   binary string  - 0x03 → 00000011
    #   bit read order - Bit 7 → 0
    #                    Bit 6 → 0
    #                    Bit 5 → 0
    #                    Bit 4 → 0
    #                    Bit 3 → 0
    #                    Bit 2 → 0
    #                    Bit 1 → 1
    #                    Bit 0 (LSB) → 1
    #   bits chunks - [5, 1, ...., 7, 3, 16]
    #   vars - ['gain', 'apd_id', ...., 'energy_step', 'priority', 'spare']
    #   unpack data - [3, 0, 0, ....., 0, 0]

    # convert data into int type for bitwise operations
    de_data = de_data.astype(np.uint64)

    for name, data in bit_structure.items():
        mask = (1 << data["bit_length"]) - 1
        unpacked[name] = de_data & mask
        # Shift the data to the right for the next iteration
        de_data = de_data >> data["bit_length"]

    return unpacked


def _create_dataset_coords(
    packets: xr.Dataset,
    apid: int,
    num_priorities: int,
    cdf_attrs: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Create the output dataset with coordinates.

    Parameters
    ----------
    packets : xarray.Dataset
        Combined packets with extracted header fields.
    apid : int
        APID for sensor type.
    num_priorities : int
        Number of priorities for this APID.
    cdf_attrs : ImapCdfAttributes
        CDF attributes manager.

    Returns
    -------
    xarray.Dataset
        Dataset with coordinates defined.
    """
    # Get timing info from the first packet of each epoch
    epoch_slice = slice(None, None, num_priorities)

    view_tab_info = ViewTabInfo(
        apid=apid,
        sensor=1 if apid == CODICEAPID.COD_HI_PHA else 0,
        collapse_table=0,
        three_d_collapsed=0,
        view_id=0,
        compression=CoDICECompression.LOSSLESS.value,  # DE data is always lossless
    )
    epochs, epochs_delta = get_codice_epoch_time(
        packets["acq_start_seconds"].isel(epoch=epoch_slice),
        packets["acq_start_subseconds"].isel(epoch=epoch_slice),
        packets["spin_period"].isel(epoch=epoch_slice),
        view_tab_info,
    )

    # Convert to numpy arrays
    epochs_data = np.asarray(epochs)
    epochs_delta_data = np.asarray(epochs_delta)
    epoch_values = met_to_ttj2000ns(epochs_data)

    dataset = xr.Dataset(
        coords={
            "epoch": (
                "epoch",
                epoch_values,
                cdf_attrs.get_variable_attributes("epoch", check_schema=False),
            ),
            "epoch_delta_minus": (
                "epoch",
                epochs_delta_data,
                cdf_attrs.get_variable_attributes(
                    "epoch_delta_minus", check_schema=False
                ),
            ),
            "epoch_delta_plus": (
                "epoch",
                epochs_delta_data,
                cdf_attrs.get_variable_attributes(
                    "epoch_delta_plus", check_schema=False
                ),
            ),
            "event_num": (
                "event_num",
                np.arange(constants.MAX_DE_EVENTS_PER_PACKET),
                cdf_attrs.get_variable_attributes("event_num", check_schema=False),
            ),
            "event_num_label": (
                "event_num",
                np.arange(constants.MAX_DE_EVENTS_PER_PACKET).astype(str),
                cdf_attrs.get_variable_attributes(
                    "event_num_label", check_schema=False
                ),
            ),
            "priority": (
                "priority",
                np.arange(num_priorities),
                cdf_attrs.get_variable_attributes("priority", check_schema=False),
            ),
            "priority_label": (
                "priority",
                np.arange(num_priorities).astype(str),
                cdf_attrs.get_variable_attributes("priority_label", check_schema=False),
            ),
        }
    )

    return dataset


def _unpack_and_store_events(
    de_data: xr.Dataset,
    packets: xr.Dataset,
    num_priorities: int,
    bit_structure: dict,
    event_fields: list[str],
) -> xr.Dataset:
    """
    Unpack all event data and store directly into the dataset arrays.

    Parameters
    ----------
    de_data : xarray.Dataset
        Dataset to store unpacked events into (modified in place).
    packets : xarray.Dataset
        Combined packets with extracted header fields.
    num_priorities : int
        Number of priorities per epoch.
    bit_structure : dict
        Bit structure defining how to unpack 64-bit event values.
    event_fields : list[str]
        List of field names to unpack (excludes priority/spare).

    Returns
    -------
    xarray.Dataset
        The dataset with unpacked events stored.
    """
    # Extract arrays from packets dataset
    num_events_arr = packets.num_events.values
    priorities_arr = packets.priority.values
    event_data_arr = packets.event_data.values

    total_events = int(np.sum(num_events_arr))
    if total_events == 0:
        return de_data

    num_packets = len(num_events_arr)

    # Preallocate arrays for concatenated events and their destination indices
    all_event_bytes = np.zeros((total_events, 8), dtype=np.uint8)
    event_epoch_idx = np.zeros(total_events, dtype=np.int32)
    event_priority_idx = np.zeros(total_events, dtype=np.int32)
    event_position_idx = np.zeros(total_events, dtype=np.int32)

    # Build concatenated event array and index mappings
    offset = 0
    for pkt_idx in range(num_packets):
        n_events = int(num_events_arr[pkt_idx])
        if n_events == 0:
            continue
        # Extract and byte-reverse events for LSB unpacking
        pkt_bytes = np.asarray(event_data_arr[pkt_idx], dtype=np.uint8)
        pkt_bytes = pkt_bytes.reshape(n_events, 8)[:, ::-1]
        all_event_bytes[offset : offset + n_events] = pkt_bytes

        # Record destination indices for later array-based assignments
        event_epoch_idx[offset : offset + n_events] = pkt_idx // num_priorities
        event_priority_idx[offset : offset + n_events] = priorities_arr[pkt_idx]
        event_position_idx[offset : offset + n_events] = np.arange(n_events)

        offset += n_events

    # Convert bytes to 64-bit values and unpack all fields at once
    all_64bits = all_event_bytes.view(np.uint64).ravel()
    unpacked = unpack_bits(bit_structure, all_64bits)

    # Place unpacked values directly into the dataset arrays
    for field in event_fields:
        de_data[field].values[
            event_epoch_idx, event_priority_idx, event_position_idx
        ] = unpacked[field]

    return de_data


def process_de_data(
    packets: xr.Dataset,
    apid: int,
    cdf_attrs: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Process direct event data into a complete CDF-ready dataset.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset containing the combined packets with extracted header fields.
    apid : int
        The APID identifying CoDICE-Lo or CoDICE-Hi.
    cdf_attrs : ImapCdfAttributes
        The CDF attributes manager.

    Returns
    -------
    xarray.Dataset
        Complete processed Direct Event dataset with coordinates and attributes.
    """
    # Get configuration for this APID
    config = constants.DE_DATA_PRODUCT_CONFIGURATIONS[apid]
    num_priorities = config["num_priorities"]
    bit_structure = config["bit_structure"]

    # Identify complete priority groups by acq_start_seconds
    # Each priority group should have exactly num_priorities packets
    # with the same acq_start_seconds value
    acq_start_seconds = packets["acq_start_seconds"].values
    unique_times, counts = np.unique(acq_start_seconds, return_counts=True)

    # Find incomplete groups (not exactly num_priorities packets)
    incomplete_mask = counts != num_priorities
    if np.any(incomplete_mask):
        incomplete_times = unique_times[incomplete_mask]
        incomplete_counts = counts[incomplete_mask]
        logger.warning(
            f"Found {len(incomplete_times)} incomplete priority group(s) "
            f"for APID {apid}. Expected {num_priorities} packets per group. "
            f"Incomplete groups at acq_start_seconds {incomplete_times.tolist()} "
            f"with counts {incomplete_counts.tolist()}. Padding with zeros."
        )

        # Create a list of groups with padding if any priorities are missing
        padded_groups = []
        for time, count in zip(unique_times, counts, strict=False):
            # Get the packets for this group
            group_ids = np.where(acq_start_seconds == time)[0]
            group = packets.isel(epoch=group_ids)
            if count < num_priorities:
                # Find missing priorities
                existing_priorities = set(group["priority"].values)
                missing_priorities = sorted(
                    set(range(num_priorities)) - existing_priorities
                )
                num_missing_priorities = len(missing_priorities)
                # Use first packet as a template and expand along the epoch dimension
                # for the number of missing priorities.
                pad_packet = group.isel(epoch=[0] * num_missing_priorities).copy()
                # Set padding values to zero
                pad_packet["num_events"].values = np.full(num_missing_priorities, 0)
                pad_packet["byte_count"].values = np.full(num_missing_priorities, 0)
                pad_packet["priority"].values = missing_priorities
                # Set event_data to empty object arrays for padding packets
                for i in range(num_missing_priorities):
                    pad_packet["event_data"].data[i] = np.array([], dtype=np.uint8)
                # Concatenate the existing priorities with the zeros priority groups
                group = xr.concat([group, pad_packet], dim="epoch")
                # Sort by priority
                sort_idx = np.argsort(group["priority"].values)
                group = group.isel(epoch=sort_idx)
            elif count > num_priorities:
                # TODO is this possible?
                # Sort by priority
                sort_idx = np.argsort(group["priority"].values)
                group = group.isel(epoch=sort_idx)
                # Keep only the first num_priorities packets
                group = group.isel(epoch=slice(0, num_priorities))
            padded_groups.append(group)

        # Concatenate all groups
        packets = xr.concat(padded_groups, dim="epoch")

    # Calculate number of epochs
    num_epochs = len(unique_times)

    # Create dataset with coordinates
    de_data = _create_dataset_coords(packets, apid, num_priorities, cdf_attrs)

    # Set global attributes based on APID
    if apid == CODICEAPID.COD_LO_PHA:
        de_data.attrs = cdf_attrs.get_global_attributes(
            "imap_codice_l1a_lo-direct-events"
        )
        de_data["k_factor"] = xr.DataArray(
            np.array([constants.K_FACTOR]),
            dims=["k_factor"],
            attrs=cdf_attrs.get_variable_attributes("k_factor", check_schema=False),
        )
    else:
        de_data.attrs = cdf_attrs.get_global_attributes(
            "imap_codice_l1a_hi-direct-events"
        )

    # Add per-epoch metadata from first packet of each epoch
    epoch_slice = slice(None, None, num_priorities)
    for var in [
        "sw_bias_gain_mode",
        "st_bias_gain_mode",
        "rgfo_esa_step",
        "rgfo_half_spin",
        "rgfo_spin_sector",
        "nso_esa_step",
        "nso_half_spin",
        "nso_spin_sector",
    ]:
        de_data[var] = xr.DataArray(
            packets[var].isel(epoch=epoch_slice).values,
            dims=["epoch"],
            attrs=cdf_attrs.get_variable_attributes(var),
        )

    # Initialize 3D event data arrays with fill values
    event_fields = [f for f in bit_structure if f not in ["priority"]]
    for field in event_fields:
        info = bit_structure[field]
        attrs = apply_replacements_to_attrs(
            cdf_attrs.get_variable_attributes("de_3d_attrs"),
            {"num_digits": len(str(info["fillval"])), "valid_max": info["fillval"]},
        )
        de_data[field] = xr.DataArray(
            np.full(
                (num_epochs, num_priorities, constants.MAX_DE_EVENTS_PER_PACKET),
                info["fillval"],
                dtype=info["dtype"],
            ),
            dims=["epoch", "priority", "event_num"],
            attrs=attrs,
        )

    # Initialize 2D per-priority metadata arrays
    for var in ["num_events", "data_quality"]:
        de_data[var] = xr.DataArray(
            np.full((num_epochs, num_priorities), 65535, dtype=np.uint16),
            dims=["epoch", "priority"],
            attrs=cdf_attrs.get_variable_attributes("de_2d_attrs"),
        )
    # Reshape packet arrays for validation and assignment
    priorities_2d = packets.priority.values.reshape(num_epochs, num_priorities)
    num_events_2d = packets.num_events.values.reshape(num_epochs, num_priorities)
    data_quality_2d = packets.suspect.values.reshape(num_epochs, num_priorities)

    # Validate each epoch has all unique priorities
    unique_counts = np.array([len(np.unique(row)) for row in priorities_2d])
    if np.any(unique_counts != num_priorities):
        bad_epoch = np.argmax(unique_counts != num_priorities)
        raise ValueError(
            f"Priority array for epoch {bad_epoch} contains "
            f"non-unique values: {priorities_2d[bad_epoch]}"
        )

    # Assign num_events and data_quality using priorities as column indices
    epoch_idx = np.arange(num_epochs)[:, np.newaxis]
    de_data["num_events"].values[epoch_idx, priorities_2d] = num_events_2d
    de_data["data_quality"].values[epoch_idx, priorities_2d] = data_quality_2d

    # Unpack all events and store directly into dataset arrays
    de_data = _unpack_and_store_events(
        de_data,
        packets,
        num_priorities,
        bit_structure,
        event_fields,
    )

    return de_data


def l1a_direct_event(unpacked_dataset: xr.Dataset, apid: int) -> xr.Dataset:
    """
    Process CoDICE L1A Direct Event data.

    Parameters
    ----------
    unpacked_dataset : xarray.Dataset
        Input L1A Direct Event dataset.
    apid : int
        APID to process.

    Returns
    -------
    xarray.Dataset
        Processed L1A Direct Event dataset.
    """
    # Combine segmented packets and extract header fields
    packets = combine_segmented_packets(
        unpacked_dataset, binary_field_name="event_data"
    )

    packets = extract_initial_items_from_combined_packets(packets)

    # Gather the CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")

    # Process packets into complete CDF-ready dataset
    return process_de_data(packets, apid, cdf_attrs)
