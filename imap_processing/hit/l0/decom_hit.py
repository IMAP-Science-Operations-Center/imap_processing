"""Decommutate HIT CCSDS science data."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.utils import packet_file_to_datasets

# Structure to hold binary details for a
# section of science data. Used to unpack
# the binary string.
HITPacking = namedtuple(
    "HITPacking",
    [
        "bit_length",
        "section_length",
        "shape",
    ],
)

# Dict of data structure for counts rates data
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
    # other rates
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

# Dict of data structure for pulse height event data
PHA_DATA_STRUCTURE = {
    # field: bit_length, section_length, shape
    "pha_records": HITPacking(2, 29344, (917,)),
}


def parse_data(
    bin_str: str, bits_per_index: int, start: int, end: int
) -> list[int] | int:
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
    parsed_data : list or int
        Integers parsed from the binary string.
    """
    parsed_data = [
        int(bin_str[i : i + bits_per_index], 2)
        for i in range(start, end, bits_per_index)
    ]
    if len(parsed_data) < 2:
        # Return value to be put in a 1D array for a science frame
        return parsed_data[0]

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
        if field == "sngrates":
            # Split into high and low gain arrays
            for i, data in enumerate(parsed_data):
                high_gain = data[::2]  # Items at even indices 0, 2, 4, etc.
                low_gain = data[1::2]  # Items at odd indices 1, 3, 5, etc.
                parsed_data[i] = [high_gain, low_gain]

        # TODO
        #  - subcommutate sectorates
        #  - decompress data
        #  - Follow up with HIT team about erates and evrates.
        #    Should these be arrays containing all the sub fields
        #    or should each subfield be it's own data field/array

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


def get_starting_packet_index(seq_flgs: xr.DataArray, start_index=0) -> int | None:
    """
    Get index of starting packet for the next science frame.

    The sequence flag of the first packet in a science frame,
    which consists of 20 packets, will have a value of 1. Given
    a starting index, this function will find the next packet
    in the dataset with a sequence flag = 1 and return that index.

    This function is used to skip invalid packets and begin
    processing the next science frame in the dataset.

    Parameters
    ----------
    seq_flgs : xr.DataArray
        Array of sequence flags in a dataset.
    start_index : int
        Starting index to find the first packet in a
        science frame from an array of sequence flags.

    Returns
    -------
    flag_index : int
        Index of starting packet for next science frame.
    """
    for flag_index, flag in enumerate(seq_flgs[start_index:]):
        if flag == 1:
            # return starting index of next science frame
            return flag_index + start_index
    return None


def is_valid_science_frame(seq_flgs: np.ndarray, src_seq_ctrs: np.ndarray) -> bool:
    """
    Check for valid science frame.

    Each science data packet has a sequence grouping flag that can equal
    0, 1, or 2. These flags are used to group 20 packets into science
    frames. This function checks if the sequence flags for a set of 20
    data packets have values that form a complete science frame.
    The first packet should have a sequence flag equal to 1. The middle
    18 packets should have a sequence flag equal to 0 and the final packet
    should have a sequence flag equal to 2.

    Valid science frame sequence flags
    [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2]

    Additionally, packets have a sequence counter. This function also
    checks if the counters are in sequential order.

    Both conditions need to be met for a science frame to be considered
    valid.

    Parameters
    ----------
    seq_flgs : numpy.ndarray
        Array of sequence flags for a set of 20 packets.
    src_seq_ctrs : numpy.ndarray
        Array of sequence counters for a set of 20 packets.

    Returns
    -------
    boolean : bool
        Boolean for whether the science frame is valid.
    """
    # Check sequence grouping flags
    if not (seq_flgs[0] == 1 and all(seq_flgs[1:19] == 0) and seq_flgs[19] == 2):
        # TODO log issue
        print(f"Invalid seq_flgs found: {seq_flgs}")
        return False

    # Check that sequence counters are sequential
    if not np.all(np.diff(src_seq_ctrs) == 1):
        # TODO log issue
        print(f"Non-sequential src_seq_ctr found: {src_seq_ctrs}")
        return False

    return True


def update_ccsds_header_data(sci_dataset: xr.Dataset) -> xr.Dataset:
    """
    Update dimensions of CCSDS header fields.

    The CCSDS header fields contain 1D arrays with
    values from all the packets in the file. This
    function updates the dimension for these fields
    to use sc_tick instead of epoch. sc_tick is the
    time the packet was created.

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
    # TODO: status bits needs to be further parsed (table 10 in algorithm doc)
    return sci_dataset


def assemble_science_frames(sci_dataset: xr.Dataset) -> xr.Dataset:
    """
    Group packets into science frames.

    HIT science frames (data from 1 minute) consist of 20 packets.
    These are assembled from the binary science_data field in the
    xarray dataset, which is a 1D array of science data from all
    packets in the file, using packet sequence grouping flags.

        First packet has a sequence flag = 1
        Next 18 packets have a sequence flag = 0
        Last packet has a sequence flag = 2

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
    #  across CCSDS files and still need to be processed. Only discard
    #  incomplete science frames in the middle of the CCSDS file.
    #  The code currently skips all incomplete science frames.

    # Initialize lists to store data from valid science frames
    count_rates_binary = []
    pha_binary = []
    epoch_science_frame = []

    science_frame_start = 0
    while science_frame_start + 20 <= len(sci_dataset.epoch):
        # Extract chunks for the current science frame (20 packets)
        seq_flgs_chunk = sci_dataset.seq_flgs[
            science_frame_start : science_frame_start + 20
        ].values
        src_seq_ctr_chunk = sci_dataset.src_seq_ctr[
            science_frame_start : science_frame_start + 20
        ].values
        science_data_chunk = sci_dataset.science_data[
            science_frame_start : science_frame_start + 20
        ]
        epoch_data_chunk = sci_dataset.epoch[
            science_frame_start : science_frame_start + 20
        ]

        if is_valid_science_frame(seq_flgs_chunk, src_seq_ctr_chunk):
            # Append valid data to lists
            count_rates_binary.append("".join(science_data_chunk.data[0:6]))
            pha_binary.append("".join(science_data_chunk.data[6:]))
            # Just take first packet's epoch for the science frame
            epoch_science_frame.append(epoch_data_chunk[0])
            science_frame_start += 20  # Move to the next science frame
        else:
            print(
                f"Invalid science frame found with starting packet index = "
                f"{science_frame_start}"
            )
            # Skip science frame and move on to the next science frame packets
            # Get index for the first packet in the next science frame
            start = get_starting_packet_index(
                sci_dataset.seq_flgs.values, start_index=science_frame_start
            )
            if start:
                science_frame_start = start
                print(
                    f"Next science frame found with starting packet index = "
                    f"{science_frame_start}"
                )
                # TODO: for skipped science frames, remove corresponding values from ccsds
                #  headers as well? Those fields contain values from all packets in a file
            else:
                # TODO raise error or log issue
                print("No other valid science frames found in the file")
                break

    # TODO: check and log if there are extra packets at end of file
    # TODO: add all dimensions to coordinates or just do that in hit_l1a.py
    #       when the cdf yaml is used to update all the dimension names?

    # Add new data variables to the dataset
    epoch_science_frame = np.array(epoch_science_frame)
    # Replace epoch per packet dimension with epoch per science frame dimension
    sci_dataset = sci_dataset.drop_vars("epoch")
    sci_dataset.coords["epoch"] = epoch_science_frame
    sci_dataset["count_rates_binary"] = xr.DataArray(
        count_rates_binary, dims=["epoch"], name="count_rates_binary"
    )
    sci_dataset["pha_binary"] = xr.DataArray(
        pha_binary, dims=["epoch"], name="pha_binary"
    )
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
    sci_dataset = update_ccsds_header_data(sci_dataset)
    # Group science packets into groups of 20
    sci_dataset = assemble_science_frames(sci_dataset)
    # Parse count rates data from binary and add to dataset
    parse_count_rates(sci_dataset)

    # TODO:
    #  Parse binary PHA data and add to dataset (function call)

    return sci_dataset


# TODO: remove main after code is finalized
if __name__ == "__main__":
    packet_definition = (
        imap_module_directory / "hit/packet_definitions/hit_packet_definitions.xml"
    )

    # L0 file path
    packet_file = Path(imap_module_directory / "tests/hit/test_data/sci_sample.ccsds")

    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file,
        xtce_packet_definition=packet_definition,
    )

    science_dataset = datasets_by_apid[1252]
    science_dataset = decom_hit(science_dataset)
    print(science_dataset)
