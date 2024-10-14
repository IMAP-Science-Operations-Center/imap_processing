"""Decommutate HIT CCSDS science data."""

from collections import namedtuple

import numpy as np
import xarray as xr

from imap_processing.utils import convert_to_binary

# TODO: Consider moving global values into a config file

# Structure to hold binary details for a
# section of science data. Used to unpack
# binary data.
HITPacking = namedtuple(
    "HITPacking",
    [
        "bit_length",
        "section_length",
        "shape",
    ],
)

# Define data structure for counts rates data
COUNTS_DATA_STRUCTURE = {
    # field: bit_length, section_length, shape
    # ------------------------------------------
    # science frame header
    "hdr_unit_num": HITPacking(2, 2, (1,)),
    "hdr_frame_version": HITPacking(6, 6, (1,)),
    "hdr_status_bits": HITPacking(8, 8, (1,)),
    "hdr_minute_cnt": HITPacking(8, 8, (1,)),
    # ------------------------------------------
    # spare bits. Contains no data
    "spare": HITPacking(24, 24, (1,)),
    # ------------------------------------------
    # erates - contains livetime counters
    "livetime": HITPacking(16, 16, (1,)),  # livetime counter
    "num_trig": HITPacking(16, 16, (1,)),  # number of triggers
    "num_reject": HITPacking(16, 16, (1,)),  # number of rejected events
    "num_acc_w_pha": HITPacking(
        16, 16, (1,)
    ),  # number of accepted events with PHA data
    "num_acc_no_pha": HITPacking(16, 16, (1,)),  # number of events without PHA data
    "num_haz_trig": HITPacking(16, 16, (1,)),  # number of triggers with hazard flag
    "num_haz_reject": HITPacking(
        16, 16, (1,)
    ),  # number of rejected events with hazard flag
    "num_haz_acc_w_pha": HITPacking(
        16, 16, (1,)
    ),  # number of accepted hazard events with PHA data
    "num_haz_acc_no_pha": HITPacking(
        16, 16, (1,)
    ),  # number of hazard events without PHA data
    # -------------------------------------------
    "sngrates": HITPacking(16, 1856, (2, 58)),  # single rates
    # -------------------------------------------
    # evprates - contains event processing rates
    "nread": HITPacking(16, 16, (1,)),  # events read from event fifo
    "nhazard": HITPacking(16, 16, (1,)),  # events tagged with hazard flag
    "nadcstim": HITPacking(16, 16, (1,)),  # adc-stim events
    "nodd": HITPacking(16, 16, (1,)),  # odd events
    "noddfix": HITPacking(16, 16, (1,)),  # odd events that were fixed in sw
    "nmulti": HITPacking(
        16, 16, (1,)
    ),  # events with multiple hits in a single detector
    "nmultifix": HITPacking(16, 16, (1,)),  # multi events that were fixed in sw
    "nbadtraj": HITPacking(16, 16, (1,)),  # bad trajectory
    "nl2": HITPacking(16, 16, (1,)),  # events sorted into L12 event category
    "nl3": HITPacking(16, 16, (1,)),  # events sorted into L123 event category
    "nl4": HITPacking(16, 16, (1,)),  # events sorted into L1423 event category
    "npen": HITPacking(16, 16, (1,)),  # events sorted into penetrating event category
    "nformat": HITPacking(16, 16, (1,)),  # nothing currently goes in this slot
    "naside": HITPacking(16, 16, (1,)),  # A-side events
    "nbside": HITPacking(16, 16, (1,)),  # B-side events
    "nerror": HITPacking(16, 16, (1,)),  # events that caused a processing error
    "nbadtags": HITPacking(
        16, 16, (1,)
    ),  # events with inconsistent tags vs pulse heights
    # -------------------------------------------
    # other count rates
    "coinrates": HITPacking(16, 416, (26,)),  # coincidence rates
    "bufrates": HITPacking(16, 512, (32,)),  # priority buffer rates
    "l2fgrates": HITPacking(16, 2112, (132,)),  # range 2 foreground rates
    "l2bgrates": HITPacking(16, 192, (12,)),  # range 2 background rates
    "l3fgrates": HITPacking(16, 2672, (167,)),  # range 3 foreground rates
    "l3bgrates": HITPacking(16, 192, (12,)),  # range 3 background rates
    "penfgrates": HITPacking(16, 528, (33,)),  # range 4 foreground rates
    "penbgrates": HITPacking(16, 240, (15,)),  # range 4 background rates
    "ialirtrates": HITPacking(16, 320, (20,)),  # ialirt rates
    "sectorates": HITPacking(16, 1920, (120,)),  # sectored rates
    "l4fgrates": HITPacking(16, 768, (48,)),  # all range foreground rates
    "l4bgrates": HITPacking(16, 384, (24,)),  # all range foreground rates
}

# Define data structure for pulse height event data
PHA_DATA_STRUCTURE = {
    # field: bit_length, section_length, shape
    "pha_records": HITPacking(2, 29344, (917,)),
}

# Define the pattern of grouping flags in a complete science frame.
FLAG_PATTERN = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2])

# Define size of science frame (num of packets)
FRAME_SIZE = len(FLAG_PATTERN)


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
    stored as count_rates_binary in the dataset,
    according to data structure details provided in
    COUNTS_DATA_STRUCTURE. The parsed data, representing
    integers, is added to the dataset as new data
    fields.

    Note: count_rates_binary is added to the dataset by
    the assemble_science_frames function, which organizes
    the binary science data packets by science frames.

    Parameters
    ----------
    sci_dataset : xr.Dataset
        Xarray dataset containing HIT science packets
        from a CCSDS file.
    """
    counts_binary = sci_dataset.count_rates_binary
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

        # TODO
        #  - status bits needs to be further parsed (table 10 in algorithm doc)
        #  - subcommutate sectorates
        #  - decompress data
        #  - Follow up with HIT team about erates and evrates.
        #    (i.e.Should these be arrays containing all the sub fields
        #    or should each subfield be it's own data field/array)

        # Get dims for data variables (yaml file not created yet)
        if len(field_meta.shape) > 1:
            dims = ["epoch", "gain", f"{field}_index"]
        elif field_meta.shape[0] > 1:
            dims = ["epoch", f"{field}_index"]
        else:
            dims = ["epoch"]

        sci_dataset[field] = xr.DataArray(parsed_data, dims=dims, name=field)
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
    return np.all(np.diff(counters) == 1)


def find_valid_starting_indices(flags: np.ndarray, counters: np.ndarray) -> np.ndarray:
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
    # TODO: consider combining functions to get valid indices to reduce
    #  code tracing

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
    valid_indices = get_valid_indices(match_indices, counters, FRAME_SIZE)
    return valid_indices


def get_valid_indices(
    indices: np.ndarray, counters: np.ndarray, size: int
) -> np.ndarray:
    """
    Get valid indices for science frames.

    Check if the packet sequence counters for the science frames
    are sequential. If they are, the science frame is valid and
    an updated array of valid indices is returned.

    Parameters
    ----------
    indices : np.ndarray
        Array of indices where the packet grouping flags match the pattern.
    counters : np.ndarray
        Array of packet sequence counters.
    size : int
        Size of science frame. 20 packets per science frame.

    Returns
    -------
    valid_indices : np.ndarray
        Array of valid indices for science frames.
    """
    # Check if the packet sequence counters are sequential by getting an array
    # of boolean values where True indicates the counters are sequential.
    sequential_check = [is_sequential(counters[idx : idx + size]) for idx in indices]
    return indices[sequential_check]


def update_ccsds_header_dims(sci_dataset: xr.Dataset) -> xr.Dataset:
    """
    Update dimensions of CCSDS header fields.

    The CCSDS header fields contain 1D arrays with
    values from all the packets in the file.
    While the epoch dimension contains time per packet,
    it will be updated later in the process to represent
    time per science frame, so another time dimension is
    needed for the ccsds header fields.This function
    updates the dimension for these fields to use sc_tick
    instead of epoch. sc_tick is the time the packet was
    created.

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

    These groups are added to the dataset as count_rates_binary
    and pha_binary.

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
    # TODO: improve this as needed
    binary_str_val = []
    for data in sci_dataset.science_data.values:
        binary_str_val.append(convert_to_binary(data))
    science_data = binary_str_val
    epoch_data = sci_dataset.epoch.values

    # Number of packets in the file
    total_packets = len(epoch_data)

    # Find starting indices for valid science frames
    starting_indices = find_valid_starting_indices(seq_flgs, seq_ctrs)

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
    epoch_per_science_frame = np.array([])
    for idx in starting_indices:
        # Data from 20 packets in a science frame
        science_data_frame = science_data[idx : idx + FRAME_SIZE]
        # First 6 packets contain count rates data in binary
        count_rates.append("".join(science_data_frame[:6]))
        # Last 14 packets contain pulse height event data in binary
        pha.append("".join(science_data_frame[6:]))
        # Get first packet's epoch for the science frame
        epoch_per_science_frame = np.append(epoch_per_science_frame, epoch_data[idx])
        # TODO: Filter ccsds header fields to only include packets from the
        #  valid science frames. Doesn't need to be grouped by frames though

    # Add new data variables to the dataset
    sci_dataset = sci_dataset.drop_vars("epoch")
    sci_dataset.coords["epoch"] = epoch_per_science_frame
    sci_dataset["count_rates_binary"] = xr.DataArray(
        count_rates, dims=["epoch"], name="count_rates_binary"
    )
    sci_dataset["pha_binary"] = xr.DataArray(pha, dims=["epoch"], name="pha_binary")
    return sci_dataset


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

    # TODO:
    #  Parse binary PHA data and add to dataset (function call)

    return sci_dataset
