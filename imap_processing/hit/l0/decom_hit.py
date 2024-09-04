"""Decommutate HIT CCSDS data."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.utils import packet_file_to_datasets

# **********************************************************************
# NOTES:
# use_derived_value boolean flag (default True) whether to use
# the derived value from the XTCE definition.
#   True to get L1B housekeeping data with engineering units.
#   False for L1A housekeeping data
# sc_tick is the time the packet was created
# Tweaked packet_file_to_datasets function to only return 40 packets
# (2 frames for testing)
# **********************************************************************

HITPacking = namedtuple(
    "HITPacking",
    [
        "bit_length",
        "section_length",
        "shape",
    ],
)

counts_data_structure = {
    # field: bit_length, section_length, shape
    # ------------------------------------------
    # science frame header
    "hdr_unit_num": HITPacking(2, 2, (1,)),
    "hdr_frame_version": HITPacking(6, 6, (1,)),
    "hdr_status_bits": HITPacking(8, 8, (1,)),
    "hdr_minute_cnt": HITPacking(8, 8, (1,)),
    # ------------------------------------------
    # spare
    "spare": HITPacking(24, 24, (1,)),
    # ------------------------------------------
    # erates
    "livetime": HITPacking(16, 16, (1,)),
    "num_trig": HITPacking(16, 16, (1,)),
    "num_reject": HITPacking(16, 16, (1,)),
    "num_acc_w_pha": HITPacking(16, 16, (1,)),
    "num_acc_no_pha": HITPacking(16, 16, (1,)),
    "num_haz_trig": HITPacking(16, 16, (1,)),
    "num_haz_reject": HITPacking(16, 16, (1,)),
    "num_haz_acc_w_pha": HITPacking(16, 16, (1,)),
    "num_haz_acc_no_pha": HITPacking(16, 16, (1,)),
    # -------------------------------------------
    "sngrates": HITPacking(16, 1856, (58, 2)),
    # -------------------------------------------
    # evrates
    "nread": HITPacking(16, 16, (1,)),
    "nhazard": HITPacking(16, 16, (1,)),
    "nadcstim": HITPacking(16, 16, (1,)),
    "nodd": HITPacking(16, 16, (1,)),
    "noddfix": HITPacking(16, 16, (1,)),
    "nmulti": HITPacking(16, 16, (1,)),
    "nmultifix": HITPacking(16, 16, (1,)),
    "nbadtraj": HITPacking(16, 16, (1,)),
    "nl2": HITPacking(16, 16, (1,)),
    "nl3": HITPacking(16, 16, (1,)),
    "nl4": HITPacking(16, 16, (1,)),
    "npen": HITPacking(16, 16, (1,)),
    "nformat": HITPacking(16, 16, (1,)),
    "naside": HITPacking(16, 16, (1,)),
    "nbside": HITPacking(16, 16, (1,)),
    "nerror": HITPacking(16, 16, (1,)),
    "nbadtags": HITPacking(16, 16, (1,)),
    # -------------------------------------------
    "coinrates": HITPacking(16, 416, (26,)),
    "bufrates": HITPacking(16, 512, (32,)),
    "l2fgrates": HITPacking(16, 2112, (132,)),
    "l2bgrates": HITPacking(16, 192, (12,)),
    "l3fgrates": HITPacking(16, 2672, (167,)),
    "l3bgrates": HITPacking(16, 192, (12,)),
    "penfgrates": HITPacking(16, 528, (33,)),
    "penbgrates": HITPacking(16, 240, (15,)),
    "ialirtrates": HITPacking(16, 320, (20,)),
    "sectorates": HITPacking(16, 1920, (120,)),
    "l4fgrates": HITPacking(16, 768, (48,)),
    "l4bgrates": HITPacking(16, 384, (24,)),
}

pha_data_structure = {
    # field: bit_length, section_length, shape
    "pha_records": HITPacking(2, 29344, (917,)),
}


def parse_data(bin_str: str, bits_per_index: int, start: int, end: int):
    """
    Parse binary data

    Parameters
    ----------
    bin_str : str
        Binary string to be unpacked
    bits_per_index : int
        Number of bits per index of the data section
    start : int
        Starting index for slicing the binary string
    end : int
        Ending index for slicing the binary string

    Returns
    -------
    parsed_data : list
        Integers parsed from the binary string

    """
    parsed_data = [
        int(bin_str[i: i + bits_per_index], 2)
        for i in range(start, end, bits_per_index)
    ]
    if len(parsed_data) < 2:
        # Return single values to be put in a single array for a science frame
        return parsed_data[0]

    return parsed_data


def parse_count_rates(dataset: xr.Dataset) -> None:
    """Parse bin of binary count rates data and update dataset"""
    counts_bin = dataset.count_rates_bin
    # initialize the starting bit for the sections of data
    section_start = 0
    variables = {}
    # for each field type in counts_data_structure
    for field, field_meta in counts_data_structure.items():
        # for each binary string decommutate the data
        section_end = section_start + field_meta.section_length
        bits_per_index = field_meta.bit_length
        parsed_data = [
            parse_data(bin_str, bits_per_index, section_start, section_end)
            for bin_str in counts_bin.values
        ]
        if field == 'sngrates':
            # split arrays into high and low gain arrays. put into a function?
            for i, data in enumerate(parsed_data):
                high_gain = data[::2]  # Items at even indices 0, 2, 4, etc.
                low_gain = data[1::2]  # Items at odd indices 1, 3, 5, etc.
                parsed_data[i] = [high_gain, low_gain]

        # TODO
        #  - subcommutate sectorates
        #  - decompress data
        #  - Check with HIT team about erates and evrates. Should these be arrays containing all the sub fields
        #    or should each subfield be it's own data field/array

        if len(field_meta.shape) > 1:
            data_shape = (len(counts_bin), field_meta.shape[1], field_meta.shape[0])  # needed for sngrates
            dims = ["epoch_frame", "gain", "index"]
        else:
            if field_meta.shape[0] > 1:
                data_shape = (len(counts_bin), field_meta.shape[0])
                dims = ["epoch_frame", f"{field}_index"]
            else:
                # shape for list of single values (science header, erates, evrates)
                data_shape = (len(counts_bin),)
                dims = ["epoch_frame"]

        # reshape the data
        shaped_data = np.array(parsed_data).reshape(data_shape)
        variables[field] = shaped_data
        science_dataset[field] = xr.DataArray(shaped_data, dims=dims, name=field)
        # increment for the start of the next section of data
        section_start += field_meta.section_length
    # for k, v in variables.items():
    #     print(f"{k}:{v}")


def get_starting_packet_index(seq_flgs: xr.DataArray, start=0) -> int:
    """
    Get index of starting packet for the next science frame

    The sequence flag of the first packet in a science frame,
    which consists of 20 packets, will have a value of 1. Given
    a starting index, this function will find the next packet
    in the dataset with a sequence flag = 1 and return that index.

    This function is used to skip invalid packets and begin
    processing the next science frame in the dataset.

    Parameters
    ----------
    seq_flgs (xr.DataArray):
        Array of sequence flags in a dataset
    start : int
        Index to start from

    Returns
    -------
    flag_index : int
        Index of starting packet in next science frame
    """

    for flag_index, flag in enumerate(seq_flgs[start:]):
        if flag == 1:
            return flag_index


def is_valid_science_frame(seq_flgs: np.ndarray, src_seq_ctrs:np.ndarray) -> bool:
    """
    Check for valid science frame.

    Each science data packet has a sequence grouping flag that can equal
    0, 1, or 2. These flags are used to group 20 packets into science
    frames. This function checks if the sequence flags for a set of 20
    data packets have values that form a complete science frame.
    The first packet should have a sequence flag equal to 1. The middle
    18 packets should have a sequence flag equal to 0 and the final packet
    should have a sequence flag equal to 2

    Valid science frame
    [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2]

    Additionally, packets have a sequence counter. This function also
    checks if the counters are in sequential order.

    Both conditions need to be met for a science frame to be considered
    valid.

    Parameters
    ----------
    seq_flgs : numpy.ndarray
        Array of sequence flags from a set of 20 packets
    src_seq_ctrs : numpy.ndarray
        Array of sequence counters from a set of 20 packets

    Returns
    -------
    boolean : bool
        Boolean for whether the science frame is valid
    """
    # Check sequence grouping flags
    if not (
            seq_flgs[0] == 1
            and all(seq_flgs[1:19] == 0)
            and seq_flgs[19] == 2
    ):
        # TODO log issue
        print(f"Invalid seq_flgs found: {seq_flgs}")
        return False

    # Check if sequence counters are sequential
    if not np.all(np.diff(src_seq_ctrs) == 1):
        # TODO log issue
        print(f"Non-sequential src_seq_ctr found: {src_seq_ctrs}")
        return False

    return True


def assemble_science_frames(sci_dataset: xr.Dataset) -> None:
    """Group packets into science frames

    HIT science frames (data from 1 minute) consist of 20 packets.
    These are assembled using packet sequence grouping flags.

        First packet has a sequence flag = 1
        Next 18 packets have a sequence flag = 0
        Last packet has a sequence flag = 2

    The science frame is further categorized into
    L1A data products -> count rates and event data.

        The first six packets contain count rates data
        The last 14 packets contain pulse height event data

    Args:
        sci_dataset (xr.Dataset): Xarray Dataset for science data
                                  APID 1252

    """
    # Initialize lists to store data from valid science frames
    count_rates_bin = []
    pha_bin = []
    epoch_science_frame = []

    i = 0
    while i + 20 <= len(sci_dataset.epoch):
        # Extract chunks for the current science frame
        seq_flgs_chunk = sci_dataset.seq_flgs[i:i + 20].values
        src_seq_ctr_chunk = sci_dataset.src_seq_ctr[i:i + 20].values
        science_data_chunk = sci_dataset.science_data[i:i + 20]
        epoch_data_chunk = sci_dataset.epoch[i:i + 20]

        if is_valid_science_frame(seq_flgs_chunk, src_seq_ctr_chunk):
            # Append valid data to lists
            count_rates_bin.append("".join(science_data_chunk.data[0:6]))
            pha_bin.append("".join(science_data_chunk.data[6:]))
            epoch_science_frame.append(epoch_data_chunk[0])
            i += 20  # Move to the next science frame
        else:
            print(f"Invalid science frame found with starting packet index = {i}")
            start = get_starting_packet_index(sci_dataset.seq_flgs.values, start=i)
            i += start

    # TODO:
    #  check and log if there are extra packets at end of file?
    #  replace epoch with epoch for each science frame? If so, need to also group
    #  CCSDS headers
    #   sc_tick
    #   version
    #   type
    #   sec_hdr_flg
    #   pkt_apid
    #   seq_flgs
    #   src_seq_ctr
    #   pkt_len

    # Convert lists to xarray DataArrays and add as new data variables to the dataset
    epoch_science_frame = np.array(epoch_science_frame)
    sci_dataset.assign_coords(epoch_frame=epoch_science_frame)
    # sci_dataset.assign_coords(epoch=epoch_science_frame)  # replace epoch?

    sci_dataset["count_rates_bin"] = xr.DataArray(
        count_rates_bin, dims=["group"], name="count_rates_bin"
    )
    sci_dataset["pha_bin"] = xr.DataArray(pha_bin, dims=["group"], name="pha_bin")


def decom_hit(sci_dataset: xr.Dataset) -> None:
    """
    Group and decode HIt science data packets

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
    binary into integers

    Parameters
    ----------
    sci_dataset: xr.Dataset
        Xarray dataset containing HIT science packets
        from a CCSDS file

    """
    # Group science packets into groups of 20
    assemble_science_frames(sci_dataset)
    # Parse count rates data from binary and add to dataset
    parse_count_rates(science_dataset)
    # TODO:
    #  Parse PHA data from binary and add to dataset (function call)


if __name__ == "__main__":
    packet_definition = (
        imap_module_directory / "hit/packet_definitions/hit_packet_definitions.xml"
    )

    # L0 file paths
    # packet_file = Path(imap_module_directory / "tests/hit/test_data/hskp_sample.ccsds")
    packet_file = Path(imap_module_directory / "tests/hit/test_data/sci_sample.ccsds")

    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file,
        xtce_packet_definition=packet_definition,
    )

    science_dataset = datasets_by_apid[1252]
    decom_hit(science_dataset)
    print(science_dataset)


