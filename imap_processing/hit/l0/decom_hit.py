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
    # "l2fgrates": HITPacking(16, 16, ()),
    # "l2bgrates": HITPacking(16, 16, ()),
    # "l3fgrates": HITPacking(16, 16, ()),
    # "l3bgrates": HITPacking(16, 16, ()),
    # "penfgrates": HITPacking(16, 16, ()),
    # "penbgrates": HITPacking(16, 16, ()),
    # "ialirtrates": HITPacking(16, 16, ()),
    # "sectorates": HITPacking(16, 16, ()),
    # "l4fgrates": HITPacking(16, 16, ()),
    # "l4bgrates": HITPacking(16, 16, ()),
}


def parse_data(bin_str: str, bits_per_index: int, start: int, end: int) -> list[int]:
    parsed_data = [
        int(bin_str[i : i + bits_per_index], 2)
        for i in range(start, end, bits_per_index)
    ]

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

        if len(field_meta.shape) > 1:
            data_shape = (len(counts_bin), field_meta.shape[0], field_meta.shape[1])
        else:
            data_shape = (len(counts_bin), field_meta.shape[0])

        # reshape the decompressed data
        shaped_data = np.array(parsed_data).reshape(data_shape)
        variables[field] = shaped_data

        # increment for the start of the next section
        section_start += field_meta.section_length
    for k, v in variables.items():
        print(f"{k}:{v}")


def assemble_science_frames(sci_dataset: xr.Dataset) -> None:
    """Group packets into science frames

    HIT science frames consist of 20 packets (data from 1 minute).
    These are assembled using packet sequence flags.

        First packet has a sequence flag = 1
        Next 18 packets have a sequence flag = 0
        Last packet has a sequence flag = 2

    The science frame is further categorized into
    L1A data products.

        The first six packets contain count rates data
        The last 14 packets contain pulse height event data

    Args:
        sci_dataset (xr.Dataset): Xarray Dataset for science data
                                  APID 1252

    """
    # Initialize lists to store data from valid science frames
    count_rates_bin = []
    pha_bin = []
    # Iterate over the dataset in chunks of 20
    for i in range(0, len(sci_dataset.epoch), 20):
        # Check if the slice length is exactly 20
        if i + 20 <= len(sci_dataset.epoch):
            seq_flgs_chunk = sci_dataset.seq_flgs[i : i + 20]
            science_data_chunk = sci_dataset.science_data[i : i + 20]
            # TODO:
            #  Add check for sequence counter to ensure it's incrementing by one
            #  (src_seq_ctr)
            #  Add handling for epoch values. Use the SC_TICK value from the first
            #  packet in the science frame

            # Check if the first packet is 1, the middle 18 packets are 0,
            # and the last packet is 2
            if (
                seq_flgs_chunk[0] == 1
                and all(seq_flgs_chunk[1:19] == 0)
                and seq_flgs_chunk[19] == 2
            ):
                # If the chunk is valid, split science data and append to lists.
                # First 6 packets are count rates.
                # Remaining 14 packets are pulse height event data
                count_rates_bin.append("".join(science_data_chunk.data[0:6]))
                pha_bin.append("".join(science_data_chunk.data[6:]))
            else:
                # TODO: decide how to handle cases when the sequence doesn't match,
                #  skip it or raise a warning/error?
                print(f"Invalid sequence found at index {i}")

    # Convert the list to an xarray DataArray and add as new data variables to the dataset
    sci_dataset["count_rates_bin"] = xr.DataArray(
        count_rates_bin, dims=["group"], name="count_rates_bin"
    )
    sci_dataset["pha_bin"] = xr.DataArray(pha_bin, dims=["group"], name="pha_bin")


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

# Group science packets into groups of 20
science_dataset = datasets_by_apid[1252]
assemble_science_frames(science_dataset)

# Parse count rates from binary data and add them to dataset
parse_count_rates(science_dataset)

# Add data variables to dataset


print(science_dataset)
