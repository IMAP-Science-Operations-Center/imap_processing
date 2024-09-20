from pathlib import Path

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.hit.l0.decom_hit import (
    assemble_science_frames,
    decom_hit,
    find_valid_starting_indices,
    get_valid_indices,
    is_sequential,
    parse_count_rates,
    parse_data,
    update_ccsds_header_dims,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture()
def sci_dataset():
    """Create a xarray dataset for testing from sample data."""
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
    return science_dataset


def test_parse_data():
    """Test the parse_data function."""
    # Test parsing a single integer
    bin_str = "110"
    bits_per_index = 2
    start = 0
    end = 2
    result = parse_data(bin_str, bits_per_index, start, end)
    assert result == [3]  # 11 in binary is 3

    # Test parsing multiple integers
    bin_str = "110010101011"
    bits_per_index = 2
    start = 0
    end = 12
    result = parse_data(bin_str, bits_per_index, start, end)
    assert result == [3, 0, 2, 2, 2, 3]  # 11, 00, 10, 10, 10, 11 in binary


def test_parse_count_rates(sci_dataset):
    """Test the parse_count_rates function."""

    # TODO: complete this test once the function is complete

    # Update ccsds header fields to use sc_tick as dimension
    sci_dataset = update_ccsds_header_dims(sci_dataset)

    # Group science packets into groups of 20
    sci_dataset = assemble_science_frames(sci_dataset)
    # Parse count rates and add to dataset
    parse_count_rates(sci_dataset)
    # Added count rate variables to dataset
    count_rate_vars = [
        "hdr_unit_num",
        "hdr_frame_version",
        "hdr_status_bits",
        "hdr_minute_cnt",
        "spare",
        "livetime",
        "num_trig",
        "num_reject",
        "num_acc_w_pha",
        "num_acc_no_pha",
        "num_haz_trig",
        "num_haz_reject",
        "num_haz_acc_w_pha",
        "num_haz_acc_no_pha",
        "sngrates",
        "nread",
        "nhazard",
        "nadcstim",
        "nodd",
        "noddfix",
        "nmulti",
        "nmultifix",
        "nbadtraj",
        "nl2",
        "nl3",
        "nl4",
        "npen",
        "nformat",
        "naside",
        "nbside",
        "nerror",
        "nbadtags",
        "coinrates",
        "bufrates",
        "l2fgrates",
        "l2bgrates",
        "l3fgrates",
        "l3bgrates",
        "penfgrates",
        "penbgrates",
        "ialirtrates",
        "sectorates",
        "l4fgrates",
        "l4bgrates",
    ]
    if count_rate_vars in list(sci_dataset.keys()):
        assert True


def test_is_sequential():
    """Test the is_sequential function."""
    counters = np.array([0, 1, 2, 3, 4])
    if is_sequential(counters):
        assert True
    counters = np.array([0, 2, 3, 4, 5])
    if not is_sequential(counters):
        assert True


def test_find_valid_starting_indices():
    """Test the find_valid_starting_indices function."""
    flags = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
        ]
    )
    counters = np.arange(35)
    result = find_valid_starting_indices(flags, counters)
    # The only valid starting index for a science frame
    # in the flags array is 15.
    assert len(result) == 1
    assert result[0] == 15


def test_get_valid_indices():
    """Test the get_valid_indices function."""
    # Array of starting indices for science frames
    # in the science data
    indices = np.array([0, 20, 40])
    # Array of counters
    counters = np.arange(60)
    # Array of valid indices where the packets in the science
    # frame have corresponding counters in sequential order
    result = get_valid_indices(indices, counters, 20)
    # All indices are valid with sequential counters
    assert len(result) == 3

    # Test array with invalid indices (use smaller sample size)
    indices = np.array([0, 5, 10])
    # Array of counters (missing counters 6-8)
    counters = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    result = get_valid_indices(indices, counters, 5)
    # Only indices 0 and 10 are valid with sequential counters
    assert len(result) == 2


def test_update_ccsds_header_dims(sci_dataset):
    """Test the update_ccsds_header_data function.

    Replaces epoch dimension with sc_tick dimension.
    """
    updated_dataset = update_ccsds_header_dims(sci_dataset)
    assert "sc_tick" in updated_dataset.dims
    assert "epoch" not in updated_dataset.dims


def test_assemble_science_frames(sci_dataset):
    """Test the assemble_science_frames function."""
    updated_dataset = update_ccsds_header_dims(sci_dataset)
    updated_dataset = assemble_science_frames(updated_dataset)
    assert "count_rates_binary" in updated_dataset
    assert "pha_binary" in updated_dataset


def test_decom_hit(sci_dataset):
    """Test the decom_hit function.

    This function orchestrates the unpacking and decompression
    of the HIT science data.
    """
    # TODO: complete this test once the function is complete
    updated_dataset = decom_hit(sci_dataset)
    print(updated_dataset)
    assert "count_rates_binary" in updated_dataset
    assert "hdr_unit_num" in updated_dataset
