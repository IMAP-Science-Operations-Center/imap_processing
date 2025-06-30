"""Decommutate HIT CCSDS science data."""

import numpy as np
import xarray as xr

from imap_processing.hit.l0.constants import (
    AZIMUTH_ANGLES,
    COUNTS_DATA_STRUCTURE,
    EXPONENT_BITS,
    FLAG_PATTERN,
    FRAME_SIZE,
    MANTISSA_BITS,
    ZENITH_ANGLES,
)
from imap_processing.utils import convert_to_binary_string


def parse_data(bin_str: str, bits_per_index: int, start: int, end: int) -> list:
    """
    Parse binary data.

    Parameters
    ----------
    bin_str : str
        Binary string to be unpacked.
    bits_per_index : int
        Number of bits per index of the data section.
    start : int
        Starting index for slicing the binary string.
    end : int
        Ending index for slicing the binary string.

    Returns
    -------
    parsed_data : list
        Integers parsed from the binary string.
    """
    parsed_data = [
        int(bin_str[i : i + bits_per_index], 2)
        for i in range(start, end, bits_per_index)
    ]
    return parsed_data


def parse_count_rates(sci_dataset: xr.Dataset) -> None:
    """
    Parse binary count rates data and update dataset.

    This function parses the binary count rates data,
    stored as count_rates_raw in the dataset,
    according to data structure details provided in
    COUNTS_DATA_STRUCTURE. The parsed data, representing
    integers, is added to the dataset as new data
    fields.

    Note: count_rates_raw is added to the dataset by
    the assemble_science_frames function, which organizes
    the binary science data packets by science frames.

    Parameters
    ----------
    sci_dataset : xr.Dataset
        Xarray dataset containing HIT science packets
        from a CCSDS file.
    """
    counts_binary = sci_dataset.count_rates_raw
    # initialize the starting bit for the sections of data
    section_start = 0
    # Decommutate binary data for each counts data field
    for field, field_meta in COUNTS_DATA_STRUCTURE.items():
        section_end = section_start + field_meta.section_length
        bits_per_index = field_meta.bit_length
        parsed_data = [
            parse_data(bin_str, bits_per_index, section_start, section_end)
            for bin_str in counts_binary.values
        ]
        if field_meta.shape[0] == 1:
            # flatten data into a 1D array
            parsed_data = list(np.array(parsed_data).flat)

        if field == "sngrates":
            # Split into high and low gain arrays
            for i, data in enumerate(parsed_data):
                high_gain = data[::2]  # Items at even indices 0, 2, 4, etc.
                low_gain = data[1::2]  # Items at odd indices 1, 3, 5, etc.
                parsed_data[i] = [high_gain, low_gain]

        # Decompress data where needed
        if all(x not in field for x in ["hdr", "spare", "pha"]):
            parsed_data = np.vectorize(decompress_rates_16_to_32)(parsed_data)

        # Get dims for data variables
        if len(field_meta.shape) > 1:
            if "sectorates" in field:
                # Reshape data into (num_frames, 8, 15) for zenith and azimuth
                # look directions
                parsed_data = np.array(parsed_data).reshape((-1, *field_meta.shape))
                # Transpose data to (num_frames, 15, 8) for flipped look directions
                parsed_data = np.transpose(parsed_data, axes=(0, 2, 1))
                dims = ["epoch", "azimuth", "zenith"]
                # Add angle values to coordinates
                sci_dataset.coords["zenith"] = xr.DataArray(
                    data=ZENITH_ANGLES,
                    dims=["zenith"],
                    name="zenith",
                )
                sci_dataset.coords["azimuth"] = xr.DataArray(
                    data=AZIMUTH_ANGLES,
                    dims=["azimuth"],
                    name="azimuth",
                )
            elif "sngrates" in field:
                dims = ["epoch", "gain", f"{field}_index"]
        elif field_meta.shape[0] > 1:
            dims = ["epoch", f"{field}_index"]
        else:
            dims = ["epoch"]

        dtype = np.uint8 if field == "hdr_dynamic_threshold_state" else np.int64
        sci_dataset[field] = xr.DataArray(
            np.array(parsed_data, dtype=dtype), dims=dims, name=field
        )
        # Add dimensions to coordinates
        for dim in dims:
            if dim not in sci_dataset.coords:
                sci_dataset.coords[dim] = xr.DataArray(
                    np.arange(sci_dataset.sizes[dim], dtype=np.int16)
                    if dim == "gain"
                    else np.arange(sci_dataset.sizes[dim], dtype=np.int32),
                    dims=[dim],
                    name=dim,
                )

        # increment the start of the next section of data to parse
        section_start += field_meta.section_length


def is_sequential(counters: np.ndarray) -> np.bool_:
    """
    Check if an array of packet sequence counters is sequential.

    Parameters
    ----------
    counters : np.ndarray
        Array of packet sequence counters.

    Returns
    -------
    bool
        True if the sequence counters are sequential, False otherwise.
    """
    return np.all((np.diff(counters) % 16384) == 1)


def get_valid_starting_indices(flags: np.ndarray, counters: np.ndarray) -> np.ndarray:
    """
    Find valid starting indices for science frames.

    This function finds the starting indices of valid science frames.
    A valid science frame has the following packet grouping flags:

            First packet: 1
            Next 18 packets: 0
            Last packet: 2

    The packet sequence counters for the identified science frames must
    be sequential. Only the starting indices of valid science frames are
    returned.

    Parameters
    ----------
    flags : np.ndarray
        Array of packet grouping flags.
    counters : np.ndarray
        Array of packet sequence counters.

    Returns
    -------
    valid_indices : np.ndarray
        Array of valid indices for science frames.
    """
    # Use sliding windows to compare segments of the array (20 packets) with the
    # pattern. This generates an array of overlapping sub-arrays, each of length
    # 20, from the flags array and is used to slide the "window" across the array
    # and compare the sub-arrays with the predefined pattern.
    windows = np.lib.stride_tricks.sliding_window_view(flags, FRAME_SIZE)
    # Find where the windows match the pattern
    matches = np.all(windows == FLAG_PATTERN, axis=1)
    # Get the starting indices of matches
    match_indices = np.where(matches)[0]
    # Filter for only indices from valid science frames with sequential counters
    sequential_check = [
        is_sequential(counters[idx : idx + FRAME_SIZE]) for idx in match_indices
    ]
    valid_indices: np.ndarray = np.array(match_indices[sequential_check], dtype=int)
    return valid_indices


def update_ccsds_header_dims(sci_dataset: xr.Dataset) -> xr.Dataset:
    """
    Update dimensions of CCSDS header fields.

    The CCSDS header fields contain 1D arrays with
    values from all the packets in the file.
    While the epoch dimension contains time per packet,
    it will be updated later in the process to represent
    time per science frame, so another time dimension is
    needed for the ccsds header fields.This function
    updates the dimension for all data vars to use sc_tick
    instead of epoch. It also temporarily sets sc_tick as the
    dimension for the epoch coordinate (to be updated later
    in the assemble_science_frames function).

    Note: sc_tick is the time the packet was created.

    Parameters
    ----------
    sci_dataset : xr.Dataset
        Xarray dataset containing HIT science packets
        from a CCSDS file.

    Returns
    -------
    sci_dataset : xr.Dataset
        Updated xarray dataset.
    """
    # sc_tick contains spacecraft time per packet
    sci_dataset.coords["sc_tick"] = sci_dataset["sc_tick"]
    sci_dataset = sci_dataset.swap_dims({"epoch": "sc_tick"})
    return sci_dataset


def assemble_science_frames(sci_dataset: xr.Dataset) -> xr.Dataset:
    """
    Group packets into science frames.

    HIT science frames (data from 1 minute) consist of 20 packets.
    These are assembled from the binary science_data field in the
    xarray dataset, which is a 1D array of science data from all
    packets in the file, by using packet grouping flags.

    The science frame is further categorized into
    L1A data products -> count rates and event data.

        The first six packets contain count rates data
        The last 14 packets contain pulse height event data

    These groups are added to the dataset as count_rates_raw
    and pha_raw.

    Parameters
    ----------
    sci_dataset : xr.Dataset
        Xarray Dataset for science data (APID 1252).

    Returns
    -------
    sci_dataset : xr.Dataset
        Updated xarray dataset with binary count rates and pulse
        height event data per valid science frame added as new
        data variables.
    """
    # TODO: Figure out how to handle partial science frames at the
    #  beginning and end of CCSDS files. These science frames are split
    #  across CCSDS files and still need to be processed with packets
    #  from the previous file. Only discard incomplete science frames
    #  in the middle of the CCSDS file. The code currently skips all
    #  incomplete science frames.

    # Convert sequence flags and counters to NumPy arrays for vectorized operations
    seq_flgs = sci_dataset.seq_flgs.values
    seq_ctrs = sci_dataset.src_seq_ctr.values
    science_data = [
        convert_to_binary_string(data) for data in sci_dataset.science_data.values
    ]
    epoch_data = sci_dataset.epoch.values

    # Number of packets in the file
    total_packets = len(epoch_data)

    # Find starting indices for valid science frames
    starting_indices = get_valid_starting_indices(seq_flgs, seq_ctrs)

    # Check for extra packets at start and end of file
    # TODO: Will need to handle these extra packets when processing multiple files
    if starting_indices[0] != 0:
        # The first science frame start index is not at the beginning of the file.
        print(
            f"{starting_indices[0]} packets at start of file belong to science frame "
            f"from previous day's ccsds file"
        )
    last_index_of_last_frame = starting_indices[-1] + FRAME_SIZE
    if last_index_of_last_frame:
        remaining_packets = total_packets - last_index_of_last_frame
        if 0 < remaining_packets < FRAME_SIZE:
            print(
                f"{remaining_packets} packets at end of file belong to science frame "
                f"from next day's ccsds file"
            )

    # Extract data per science frame and organize by L1A data products
    count_rates = []
    pha = []
    epoch_per_science_frame = []
    for idx in starting_indices:
        # Data from 20 packets in a science frame
        science_data_frame = science_data[idx : idx + FRAME_SIZE]
        # First 6 packets contain count rates data in binary
        count_rates.append("".join(science_data_frame[:6]))
        # Last 14 packets contain pulse height event data in binary
        pha.append("".join(science_data_frame[6:]))
        # Get the mean epoch in the frame to use as the data collection time
        epoch_per_science_frame.append(
            calculate_epoch_mean(epoch_data, idx, FRAME_SIZE)
        )

    # Add new data variables to the dataset and update epoch coordinate
    sci_dataset.coords["epoch"] = xr.DataArray(
        np.array(epoch_per_science_frame, dtype=np.int64), dims=["epoch"]
    )
    sci_dataset["count_rates_raw"] = xr.DataArray(
        count_rates, dims=["epoch"], name="count_rates_raw"
    )
    sci_dataset["pha_raw"] = xr.DataArray(pha, dims=["epoch"], name="pha_raw")
    return sci_dataset


def decompress_rates_16_to_32(packed: int) -> int:
    """
    Will decompress rates data from 16 bits to 32 bits.

    This function decompresses the rates data from 16-bit integers
    to 32-bit integers. The compressed integer (packed) combines
    two parts:

    1. Mantissa: Represents the significant digits of the value.
    2. Exponent: Determines how much to scale the mantissa (using powers of 2).

    These parts are packed together into a single 16-bit integer.

    Parameters
    ----------
    packed : int
        Compressed 16-bit integer.

    Returns
    -------
    decompressed_int : int
        Decompressed integer.
    """
    # In compressed formats, the exponent and mantissa are tightly packed together.
    # The mask ensures you correctly separate the mantissa (useful for reconstructing
    # the value) from the exponent (used for scaling).
    # set to 16 bits
    output_mask = 0xFFFF

    # Packed is the compressed integer
    # Right bit shift to get the exponent
    power = packed >> MANTISSA_BITS

    # Decompress the data depending on the value of the exponent
    # If the exponent (power) extracted from the packed 16-bit integer is greater
    # than 1, the compressed value needs to be decompressed by reconstructing the
    # integer using the mantissa and exponent. If the condition is false, the
    # compressed and uncompressed values are considered the same.
    decompressed_int: int
    if power > 1:
        # Retrieve the "mantissa" portion of the packed value by masking out the
        # exponent bits
        mantissa_mask = output_mask >> EXPONENT_BITS
        mantissa = packed & mantissa_mask

        # Shift the mantissa to the left by 1 to account for the hidden bit
        # (always set to 1)
        mantissa_with_hidden_bit = mantissa | (0x0001 << MANTISSA_BITS)

        # Scale the mantissa by the exponent by shifting it to the left by (power - 1)
        decompressed_int = mantissa_with_hidden_bit << (power - 1)
    else:
        # The compressed and uncompressed values are the same
        decompressed_int = packed

    return decompressed_int


def calculate_epoch_mean(
    epoch_data: np.ndarray, idx: int, frame_size: int
) -> np.floating:
    """
    Calculate the mean epoch for a science frame.

    This function is used to get the center collection time for science data.

    Parameters
    ----------
    epoch_data : np.ndarray
        Array of epoch values for every science packet.
    idx : int
        Starting index of the science frame.
    frame_size : int
        Number of packets in the science frame.

    Returns
    -------
    float
        Mean epoch value for the science frame.
    """
    return np.mean([epoch_data[idx], epoch_data[idx + frame_size - 1]])


def decom_hit(sci_dataset: xr.Dataset) -> xr.Dataset:
    """
    Group and decode HIT science data packets.

    This function updates the science dataset with
    organized, decommutated, and decompressed data.

    The dataset that is passed in contains the unpacked
    CCSDS header and the science data as bytes as follows:

    <xarray.Dataset>
    Dimensions:       epoch
    Coordinates:
      * epoch         (epoch) int64
    Data variables:
        sc_tick       (epoch) uint32
        science_data  (epoch) <U2096
        version       (epoch) uint8
        type          (epoch) uint8
        sec_hdr_flg   (epoch) uint8
        pkt_apid      (epoch) uint16
        seq_flgs      (epoch) uint8
        src_seq_ctr   (epoch) uint16
        pkt_len       (epoch) uint16

    The science data for a science frame (i.e. 1 minute of data)
    is spread across 20 packets. This function groups the
    data into science frames and decommutates and decompresses
    binary into integers.

    Parameters
    ----------
    sci_dataset : xr.Dataset
        Xarray dataset containing HIT science packets
        from a CCSDS file.

    Returns
    -------
    sci_dataset : xr.Dataset
        Updated xarray dataset with new fields for all count
        rates and pulse height event data per valid science frame
        needed for creating an L1A product.
    """
    # Update ccsds header fields to use sc_tick as dimension
    sci_dataset = update_ccsds_header_dims(sci_dataset)

    # Group science packets into groups of 20
    sci_dataset = assemble_science_frames(sci_dataset)

    # Parse count rates data from binary and add to dataset
    parse_count_rates(sci_dataset)

    # Remove raw binary data and unused spare bits from dataset
    sci_dataset = sci_dataset.drop_vars(["count_rates_raw", "science_data", "spare"])

    return sci_dataset
