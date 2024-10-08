"""Processing function for Lo Science Data."""

from collections import namedtuple

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l0.decompression_tables.decompression_tables import (
    CASE_DECODER,
    DATA_BITS,
    DE_BIT_SHIFT,
)
from imap_processing.lo.l0.utils.bit_decompression import (
    DECOMPRESSION_TABLES,
    Decompress,
    decompress_int,
)

HistPacking = namedtuple(
    "HistPacking",
    [
        "bit_length",
        "section_length",
        "shape",  # (azimuth, esa_step)
    ],
)

HIST_DATA_META = {
    # field: bit_length, section_length, shape
    "start_a": HistPacking(12, 504, (6, 7)),
    "start_c": HistPacking(12, 504, (6, 7)),
    "stop_b0": HistPacking(12, 504, (6, 7)),
    "stop_b3": HistPacking(12, 504, (6, 7)),
    "tof0_count": HistPacking(8, 336, (6, 7)),
    "tof1_count": HistPacking(8, 336, (6, 7)),
    "tof2_count": HistPacking(8, 336, (6, 7)),
    "tof3_count": HistPacking(8, 336, (6, 7)),
    "tof0_tof1": HistPacking(8, 3360, (60, 7)),
    "tof0_tof2": HistPacking(8, 3360, (60, 7)),
    "tof1_tof2": HistPacking(8, 3360, (60, 7)),
    "silver": HistPacking(8, 3360, (60, 7)),
    "disc_tof0": HistPacking(8, 336, (6, 7)),
    "disc_tof1": HistPacking(8, 336, (6, 7)),
    "disc_tof2": HistPacking(8, 336, (6, 7)),
    "disc_tof3": HistPacking(8, 336, (6, 7)),
    "pos0": HistPacking(12, 504, (6, 7)),
    "pos1": HistPacking(12, 504, (6, 7)),
    "pos2": HistPacking(12, 504, (6, 7)),
    "pos3": HistPacking(12, 504, (6, 7)),
    "hydrogen": HistPacking(8, 3360, (60, 7)),
    "oxygen": HistPacking(8, 3360, (60, 7)),
}


def parse_histogram(dataset: xr.Dataset, attr_mgr: ImapCdfAttributes) -> xr.Dataset:
    """
    Parse and decompress binary histogram data for Lo.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science counts from packets_to_dataset function.
    attr_mgr : ImapCdfAttributes
        CDF attribute manager for Lo L1A.

    Returns
    -------
    dataset : xr.Dataset
        Parsed and decompressed histogram data.
    """
    hist_bin = dataset.sci_cnt

    # initialize the starting bit for the sections of data
    section_start = 0
    # for each field type in the histogram data
    for field in HIST_DATA_META:
        data_meta = HIST_DATA_META[field]
        # for each histogram binary string decompress
        # the data
        decompressed_data = [
            decompress(
                bin_str, data_meta.bit_length, section_start, data_meta.section_length
            )
            for bin_str in hist_bin.values
        ]

        # add on the epoch length (equal to number of packets) to the
        # field shape
        data_shape = (len(hist_bin), data_meta.shape[0], data_meta.shape[1])

        # get the dimension names from the CDF attr manager
        dims = [
            value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        ]
        # reshape the decompressed data
        shaped_data = np.array(decompressed_data, dtype=np.uint32).reshape(data_shape)
        # add the data to the dataset
        dataset[field] = xr.DataArray(
            shaped_data, dims=dims, attrs=attr_mgr.get_variable_attributes(field)
        )

        # increment for the start of the next section
        section_start += data_meta.section_length

    return dataset


def decompress(
    bin_str: str, bits_per_index: int, section_start: int, section_length: int
) -> list[int]:
    """
    Parse and decompress binary histogram data for Lo.

    Parameters
    ----------
    bin_str : str
        Binary string to decompress.
    bits_per_index : int
        Number of bits per index of the data section.
    section_start : int
        The start bit for the section of data.
    section_length : int
        The length of the section of data.

    Returns
    -------
    decompressed_ints : list[int]
        Decompressed integers for the data section.
    """
    # select the decompression method based on the bit length
    # of the compressed data
    if bits_per_index == 8:
        decompress = Decompress.DECOMPRESS8TO16
    elif bits_per_index == 12:
        decompress = Decompress.DECOMPRESS12TO16
    else:
        raise ValueError(f"Invalid bits_per_index: {bits_per_index}")

    # parse the binary and convert to integers
    raw_ints = [
        int(bin_str[i : i + bits_per_index], 2)
        for i in range(section_start, section_start + section_length, bits_per_index)
    ]

    # decompress raw integers
    decompressed_ints: list[int] = decompress_int(
        raw_ints,
        decompress,
        DECOMPRESSION_TABLES,
    )

    return decompressed_ints


def parse_events(dataset: xr.Dataset, attr_mgr: ImapCdfAttributes) -> xr.Dataset:
    """
    Parse and decompress binary direct event data for Lo.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science direct events from packets_to_dataset function.
    attr_mgr : ImapCdfAttributes
        CDF attribute manager for Lo L1A.

    Returns
    -------
    dataset : xr.Dataset
        Parsed and decompressed direct event data.
    """
    # TODO: Add logging. Want to wait until I have a better understanding of how the
    #  DEs spread across multiple packets will work first

    # Sum each count to get the total number of direct events for the pointing
    num_de: int = np.sum(dataset["count"].values)

    de_fields = [
        "coincidence_type",
        "de_time",
        "esa_step",
        "mode",
        "tof0",
        "tof1",
        "tof2",
        "tof3",
        "cksm",
        "pos",
    ]
    # Initialize all Direct Event fields with their fill value
    # L1A Direct event data will not be tied to an epoch
    # data will use a direct event index for the pointing as its coordinate/dimension
    for field in de_fields:
        dataset[field] = xr.DataArray(
            np.full(num_de, attr_mgr.get_variable_attributes(field)["FILLVAL"]),
            dims="direct_events",
        )

    # The DE index for the entire pointing
    pointing_de = 0
    # for each direct event packet in the pointing
    for pkt_idx, de_count in enumerate(dataset["count"].values):
        # initialize the bit position for the packet
        dataset.attrs["bit_pos"] = 0
        # for each direct event in the packet
        for _ in range(de_count):
            # Parse the fixed fields for the direct event
            # Coincidence Type, Time, ESA Step, Mode
            dataset = parse_fixed_fields(dataset, pkt_idx, pointing_de)
            # Parse the variable fields for the direct event
            # TOF0, TOF1, TOF2, TOF3, Checksum, Position
            dataset = parse_variable_fields(dataset, pkt_idx, pointing_de, DE_BIT_SHIFT)

            pointing_de += 1

    return dataset


def parse_fixed_fields(
    dataset: xr.Dataset, pkt_idx: int, pointing_de: int
) -> xr.Dataset:
    """
    Parse the fixed fields for a direct event.

    Fixed fields are the fields that are always transmitted for
    a direct event. These fields are the Coincidence Type,
    Time, ESA Step, and Mode.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science direct events from packets_to_dataset function.
    pkt_idx : int
        Index of the packet for the pointing.
    pointing_de : int
        Index of the total direct event for the pointing.

    Returns
    -------
    dataset : xr.Dataset
        Updated dataset with the fixed fields parsed.
    """
    # Parse the first 4 bits of the DE data for the coincidence type
    dataset["coincidence_type"].values[pointing_de] = parse_de_bin(
        dataset, pkt_idx, DATA_BITS.ABSENT
    )
    dataset.attrs["bit_pos"] += DATA_BITS.ABSENT
    # Parse the next set of bits for the direct event time
    # direct event time: time of direct event relative to start of spin
    dataset["de_time"].values[pointing_de] = parse_de_bin(
        dataset, pkt_idx, DATA_BITS.DE_TIME
    )
    dataset.attrs["bit_pos"] += DATA_BITS.DE_TIME
    # parse the next set of bits for the ESA Step
    # ESA Step: Energy step from 1 to 7
    dataset["esa_step"].values[pointing_de] = parse_de_bin(
        dataset, pkt_idx, DATA_BITS.ESA_STEP
    )
    dataset.attrs["bit_pos"] += DATA_BITS.ESA_STEP
    # parse the next set of bits for the mode
    # Mode: Energy stepping mode 0 or 1
    # this is used along with the coincidence type to determine which
    # TOF fields are transmitted
    dataset["mode"].values[pointing_de] = parse_de_bin(dataset, pkt_idx, DATA_BITS.MODE)
    dataset.attrs["bit_pos"] += DATA_BITS.MODE

    return dataset


def parse_variable_fields(
    dataset: xr.Dataset, pkt_idx: int, pointing_de: int, bit_shift: int
) -> xr.Dataset:
    """
    Parse the variable fields for a direct event.

    Variable fields are the fields that are not always transmitted.
    Which fields are transmitted is determined by the Coincidence
    type and Mode. These fields are TOF0, TOF1, TOF2, TOF3, Checksum,
    and Position. All of these fields except for Position are bit
    shifted to the right by 1 when packed into the CCSDS packets.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science direct events from packets_to_dataset function.
    pkt_idx : int
        Index of the packet for the pointing.
    pointing_de : int
        Index of the total direct event for the pointing.
    bit_shift : int
        Number of bits to shift the field to the left.

    Returns
    -------
    dataset : xr.Dataset
        Updated dataset with the fixed fields parsed.
    """
    # The decoder defines which TOF fields are
    # transmitted for this case and mode
    case_decoder = CASE_DECODER[
        (
            dataset["coincidence_type"].values[pointing_de],
            dataset["mode"].values[pointing_de],
        )
    ]

    # Check which TOF fields should have been transmitted for this
    # case number / mode combination and decompress them.
    if case_decoder.TOF0:
        dataset["tof0"].values[pointing_de] = parse_de_bin(
            dataset, pkt_idx, DATA_BITS.TOF0, bit_shift
        )
        dataset.attrs["bit_pos"] += DATA_BITS.TOF0
    if case_decoder.TOF1:
        dataset["tof1"].values[pointing_de] = parse_de_bin(
            dataset, pkt_idx, DATA_BITS.TOF1, bit_shift
        )
        dataset.attrs["bit_pos"] += DATA_BITS.TOF1
    if case_decoder.TOF2:
        dataset["tof2"].values[pointing_de] = parse_de_bin(
            dataset, pkt_idx, DATA_BITS.TOF2, bit_shift
        )
        dataset.attrs["bit_pos"] += DATA_BITS.TOF2
    if case_decoder.TOF3:
        dataset["tof3"].values[pointing_de] = parse_de_bin(
            dataset, pkt_idx, DATA_BITS.TOF3, bit_shift
        )
        dataset.attrs["bit_pos"] += DATA_BITS.TOF3
    if case_decoder.CKSM:
        dataset["cksm"].values[pointing_de] = parse_de_bin(
            dataset, pkt_idx, DATA_BITS.CKSM, bit_shift
        )
        dataset.attrs["bit_pos"] += DATA_BITS.CKSM
    if case_decoder.POS:
        dataset["pos"].values[pointing_de] = parse_de_bin(
            dataset, pkt_idx, DATA_BITS.POS
        )
        dataset.attrs["bit_pos"] += DATA_BITS.POS

    return dataset


def parse_de_bin(
    dataset: xr.Dataset, pkt_idx: int, bit_length: int, bit_shift: int = 0
) -> int:
    """
    Parse a binary string for a direct event field.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science direct events from packets_to_dataset function.
    pkt_idx : int
        Index of the packet for the pointing.
    bit_length : int
        Length of the field in bits.
    bit_shift : int
        Number of bits to shift the field to the left.

    Returns
    -------
    int
        Parsed integer for the direct event field.
    """
    bit_pos = dataset.attrs["bit_pos"]
    parsed_int = (
        int(
            dataset["data"].values[pkt_idx][bit_pos : bit_pos + bit_length],
            2,
        )
        << bit_shift
    )
    return parsed_int
