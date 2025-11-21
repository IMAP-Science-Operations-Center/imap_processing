"""Processing function for Lo Science Data."""

import logging
from collections import namedtuple

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.l0.decom_hit import is_sequential
from imap_processing.lo.l0.decompression_tables.decompression_tables import (
    CASE_DECODER,
    DE_BIT_SHIFT,
    FIXED_FIELD_BITS,
    PACKET_FIELD_BITS,
    VARIABLE_FIELD_BITS,
)
from imap_processing.lo.l0.utils.bit_decompression import (
    DECOMPRESSION_TABLES,
    Decompress,
    decompress_int,
)
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.utils import convert_to_binary_string

logger = logging.getLogger(__name__)

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
    "start_a": HistPacking(12, 504, (7, 6)),
    "start_c": HistPacking(12, 504, (7, 6)),
    "stop_b0": HistPacking(12, 504, (7, 6)),
    "stop_b3": HistPacking(12, 504, (7, 6)),
    "tof0_count": HistPacking(8, 336, (7, 6)),
    "tof1_count": HistPacking(8, 336, (7, 6)),
    "tof2_count": HistPacking(8, 336, (7, 6)),
    "tof3_count": HistPacking(8, 336, (7, 6)),
    "tof0_tof1": HistPacking(8, 3360, (7, 60)),
    "tof0_tof2": HistPacking(8, 3360, (7, 60)),
    "tof1_tof2": HistPacking(8, 3360, (7, 60)),
    "silver": HistPacking(8, 3360, (7, 60)),
    "disc_tof0": HistPacking(8, 336, (7, 6)),
    "disc_tof1": HistPacking(8, 336, (7, 6)),
    "disc_tof2": HistPacking(8, 336, (7, 6)),
    "disc_tof3": HistPacking(8, 336, (7, 6)),
    "pos0": HistPacking(12, 504, (7, 6)),
    "pos1": HistPacking(12, 504, (7, 6)),
    "pos2": HistPacking(12, 504, (7, 6)),
    "pos3": HistPacking(12, 504, (7, 6)),
    "hydrogen": HistPacking(8, 3360, (7, 60)),
    "oxygen": HistPacking(8, 3360, (7, 60)),
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
    hist_bin = [convert_to_binary_string(data) for data in dataset.sci_cnt.values]

    # initialize the starting bit for the sections of data
    section_start = 0
    # for each field type in the histogram data
    for field, data_meta in HIST_DATA_META.items():
        # for each histogram binary string decompress
        # the data
        decompressed_data = [
            decompress(
                bin_str, data_meta.bit_length, section_start, data_meta.section_length
            )
            for bin_str in hist_bin
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

    This function works directly with raw bytes instead of converting to binary strings,
    resulting in significant performance improvements.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science direct events from packets_to_dataset function.
        Should contain raw bytes data in 'data' field.
    attr_mgr : ImapCdfAttributes
        CDF attribute manager for Lo L1A.

    Returns
    -------
    dataset : xr.Dataset
        Parsed and decompressed direct event data.
    """
    logger.info("\n Parsing Lo L1A Direct Events")
    # Sum each count to get the total number of direct events for the pointing
    # parse the count and passes fields. These fields only occur once
    # at the beginning of each packet group and are not part of the
    # compressed direct event data

    # Extract DE counts from raw bytes
    de_counts = [
        extract_bits_from_bytes(raw_data, 0, 16) for raw_data in dataset["data"].values
    ]

    dataset["de_count"] = xr.DataArray(
        de_counts,
        dims="epoch",
        attrs=attr_mgr.get_variable_attributes("de_count"),
    )
    num_de: int = np.sum(dataset["de_count"].values)

    logger.info(f"Total number of direct events in this ASC: {num_de}")

    de_fields = (
        list(PACKET_FIELD_BITS._asdict().keys())
        + list(FIXED_FIELD_BITS._asdict().keys())
        + list(VARIABLE_FIELD_BITS._asdict().keys())
    )

    # Initialize all Direct Event fields with their fill value
    # L1A Direct event data will not be tied to an epoch
    # data will use a direct event index for the
    # pointing as its coordinate/dimension
    for field in de_fields:
        dataset[field] = xr.DataArray(
            np.full(num_de, attr_mgr.get_variable_attributes(field)["FILLVAL"]),
            dims="direct_events",
            attrs=attr_mgr.get_variable_attributes(field),
        )
    dataset["passes"] = xr.DataArray(
        np.full(
            len(dataset["data"].values),
            attr_mgr.get_variable_attributes("passes")["FILLVAL"],
        ),
        dims="epoch",
        attrs=attr_mgr.get_variable_attributes("passes"),
    )

    # Pre-extract numpy arrays for all fields to avoid xarray overhead
    field_arrays = {}
    for field in de_fields:
        field_arrays[field] = dataset[field].values

    data_values = dataset["data"].values
    de_count_values = dataset["de_count"].values
    passes_values = dataset["passes"].values

    # Process each packet
    pointing_de = 0

    for pkt_idx, de_count in enumerate(de_count_values):
        logger.debug(
            f"Parsing packet {pkt_idx} of {len(de_count_values)} "
            f"with {de_count} direct events"
        )
        raw_data = data_values[pkt_idx]

        # Parse all direct events in this packet using bytewise operations
        pointing_de, passes_value = parse_packet_events(
            raw_data, de_count, pointing_de, field_arrays
        )
        passes_values[pkt_idx] = passes_value

    logger.info("\n Returning Lo L1A Direct Events Dataset")
    return dataset


def parse_packet_events(
    raw_data: bytes, de_count: int, pointing_de: int, field_arrays: dict
) -> tuple[int, int]:
    """
    Parse all direct events in a single packet using bitwise operations on raw bytes.

    Parameters
    ----------
    raw_data : bytes
        Raw packet data as bytes.
    de_count : int
        Number of direct events in this packet.
    pointing_de : int
        Starting index for direct events in the pointing.
    field_arrays : dict
        Dictionary of field names to pre-extracted numpy arrays.

    Returns
    -------
    int, int
        Updated pointing_de index after processing all events in packet.
        Passes value for this packet.
    """
    # Parse passes field (bits 16-47)
    passes_value = extract_bits_from_bytes(raw_data, 16, 32)

    bit_offset = 48  # Start after count (16 bits) + passes (32 bits)

    # Process all direct events in this packet
    for de_idx in range(de_count):
        current_de_idx = pointing_de + de_idx

        # Parse fixed fields using bitwise operations
        for field, bit_length in FIXED_FIELD_BITS._asdict().items():
            field_arrays[field][current_de_idx] = extract_bits_from_bytes(
                raw_data, bit_offset, bit_length
            )
            bit_offset += bit_length

        # Parse variable fields based on coincidence type and mode
        # Variable fields are the fields that are not always transmitted.
        # Which fields are transmitted is determined by the Coincidence
        # type and Mode. These fields are TOF0, TOF1, TOF2, TOF3, Checksum,
        # and Position. All of these fields except for Position are bit
        # shifted to the right by 1 when packed into the CCSDS packets.
        case_decoder = CASE_DECODER[
            (
                field_arrays["coincidence_type"][current_de_idx],
                field_arrays["mode"][current_de_idx],
            )
        ]

        for field, field_exists in case_decoder._asdict().items():
            if field_exists:
                bit_length = VARIABLE_FIELD_BITS._asdict()[field]
                bit_shift = DE_BIT_SHIFT.get(field, 0)
                field_arrays[field][current_de_idx] = extract_bits_from_bytes(
                    raw_data, bit_offset, bit_length, bit_shift
                )
                bit_offset += bit_length

    return pointing_de + de_count, passes_value


def extract_bits_from_bytes(
    data: bytes, bit_offset: int, bit_length: int, bit_shift: int = 0
) -> int:
    """
    Extract bits from raw bytes using bitwise operations.

    This is much faster than converting to binary strings and doing string slicing.

    Parameters
    ----------
    data : bytes
        Raw byte data.
    bit_offset : int
        Starting bit position (0-based).
    bit_length : int
        Number of bits to extract.
    bit_shift : int
        Number of bits to shift result left (for unpacking compressed data).

    Returns
    -------
    int
        Extracted value.
    """
    # Convert bytes to a big integer for bit manipulation
    total_bits = len(data) * 8
    value = int.from_bytes(data, byteorder="big")

    # Create a mask for the desired bits
    mask = (1 << bit_length) - 1

    # Shift to align the desired bits to the right, then apply mask
    shift_amount = total_bits - bit_offset - bit_length
    extracted = (value >> shift_amount) & mask

    # Apply any additional bit shift for decompression
    return extracted << bit_shift


def combine_segmented_packets(dataset: xr.Dataset) -> xr.Dataset:
    """
    Combine segmented packets and set MET field.

    If the number of bits needed to pack the direct events exceeds the
    maximum number of bits allowed in a packet, the direct events
    will be spread across multiple packets. This function will combine
    the segmented binary into a single binary string for each epoch.

    This function also sets the MET field based on segment start times,
    even when no segmentation is present.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo science direct events from packets_to_dataset function.

    Returns
    -------
    dataset : xr.Dataset
        Updated dataset with any segmented direct events combined and MET field set.
    """
    seq_flgs = dataset.seq_flgs.values
    seq_ctrs = dataset.src_seq_ctr.values

    # Find the start and end of each segment of direct events
    # 1 = start of a group of segmented packet
    # 2 = end of a group of segmented packets
    # 3 = unsegmented packet
    seg_starts = np.nonzero((seq_flgs == 1) | (seq_flgs == 3))[0]
    seg_ends = np.nonzero((seq_flgs == 2) | (seq_flgs == 3))[0]

    # Swap the epoch dimension for the shcoarse
    # the epoch dimension will be reduced to the
    # first epoch in each segment
    dataset.coords["shcoarse"] = dataset["shcoarse"]
    dataset = dataset.swap_dims({"epoch": "shcoarse"})

    # Find the valid groups of segmented packets
    valid_groups = find_valid_groups(seq_ctrs, seg_starts, seg_ends)

    # Combine the segmented packets into raw bytes directly
    combined_data_list = []
    for start, end in zip(seg_starts, seg_ends, strict=False):
        combined_bytes = b"".join(dataset["data"].values[start : end + 1])
        combined_data_list.append(combined_bytes)

    # Drop any group of segmented packets that aren't sequential
    valid_combined_data = [
        combined_data_list[i] for i, valid in enumerate(valid_groups) if valid
    ]

    # Update the epoch to the first epoch in the segment
    dataset.coords["epoch"] = dataset["epoch"].values[seg_starts]
    # Drop any group of segmented epochs that aren't sequential
    dataset.coords["epoch"] = dataset["epoch"].values[valid_groups]

    # Create the data DataArray with combined raw bytes
    dataset["data"] = xr.DataArray(
        valid_combined_data,
        dims=["epoch"],
        coords={"epoch": dataset.coords["epoch"]},
    )
    # Set met to the first segment start times for the valid groups
    dataset["met"] = xr.DataArray(
        dataset["shcoarse"].values[seg_starts][valid_groups], dims="epoch"
    )

    return dataset


def find_valid_groups(
    seq_ctrs: np.ndarray, seg_starts: np.ndarray, seg_ends: np.ndarray
) -> list[np.bool_]:
    """
    Find the valid groups of segmented packets.

    Parameters
    ----------
    seq_ctrs : np.ndarray
        Sequence counters from the CCSDS header.
    seg_starts : np.ndarray
        Start index of each group of segmented direct event packet.
    seg_ends : np.ndarray
        End index of each group of segmented direct event packet.

    Returns
    -------
    valid_groups : list[np.bool_]
        Valid groups of segmented packets.
    """
    # Check if the sequence counters from the CCSDS header are sequential
    grouped_seq_ctrs = [
        np.array(seq_ctrs[start : end + 1])
        for start, end in zip(seg_starts, seg_ends, strict=False)
    ]
    valid_groups = [is_sequential(seq_ctrs) for seq_ctrs in grouped_seq_ctrs]
    return valid_groups


def organize_spin_data(dataset: xr.Dataset, attr_mgr: ImapCdfAttributes) -> xr.Dataset:
    """
    Organize the spin data for Lo.

    The spin data is spread across 28 fields. This function
    combines each of those fields into 2D arrays for each
    epoch and spin.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo spin data from packets_to_dataset function.
    attr_mgr : ImapCdfAttributes
        CDF attribute manager for Lo L1A.

    Returns
    -------
    dataset : xr.Dataset
        Updated dataset with the spin data organized.
    """
    # Get the spin data fields
    spin_fields = [
        "start_sec_spin",
        "start_subsec_spin",
        "esa_neg_dac_spin",
        "esa_pos_dac_spin",
        "valid_period_spin",
        "valid_phase_spin",
        "period_source_spin",
    ]

    # Set epoch to the acq_start time
    # acq_start_sec is in units of seconds
    # acq_start_subsec is in units of microseconds
    acq_start = dataset.acq_start_sec + (1e-6 * dataset.acq_start_subsec)
    epoch = met_to_ttj2000ns(acq_start)
    dataset = dataset.assign_coords(epoch=("epoch", epoch))
    for spin_field in spin_fields:
        # Get the field attributes
        field_attrs = attr_mgr.get_variable_attributes(spin_field, check_schema=False)
        dtype = field_attrs.pop("dtype")

        packet_fields = [f"{spin_field}_{i}" for i in range(1, 29)]
        # Combine the spin data fields along a new dimension
        combined_spin_data = xr.concat(
            [dataset[field].astype(dtype) for field in packet_fields], dim="spin"
        )

        # Assign the combined data back to the dataset
        dataset[spin_field] = xr.DataArray(
            combined_spin_data.transpose(),
            dims=["epoch", "spin"],
            attrs=field_attrs,
        )
        # Drop the individual spin data fields
        dataset = dataset.drop_vars(packet_fields)

    return dataset
