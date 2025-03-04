import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.parse_mag import (
    calculate_time,
    filter_valid_groups,
    find_groups,
    get_bytes,
    get_pkt_counter,
    get_status_data,
    get_time,
    parse_packet,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_mag_path():
    """Returns the xtce directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_mag.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the paths to the binary packets."""
    directory = imap_module_directory / "tests" / "ialirt" / "test_data" / "l0"
    filenames = [
        "461971383-404.bin",
        "461971384-405.bin",
        "461971385-406.bin",
        "461971386-407.bin",
        "461971387-408.bin",
        "461971388-409.bin",
        "461971389-410.bin",
        "461971390-411.bin",
        "461971391-412.bin",
    ]
    return tuple(directory / fname for fname in filenames)


@pytest.fixture(scope="session")
def mag_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "sample_decoded_i-alirt_data.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture()
def xarray_data(binary_packet_path, xtce_mag_path):
    """Create xarray data for multiple packets."""
    apid = 1001

    xarray_data = tuple(
        packet_file_to_datasets(packet, xtce_mag_path, use_derived_value=False)[apid]
        for packet in binary_packet_path
    )

    merged_xarray_data = xr.concat(xarray_data, dim="epoch")
    return merged_xarray_data


@pytest.fixture()
def grouped_data():
    """Creates grouped data for filter_valid_groups test."""
    epoch = np.arange(12)

    # Example `src_seq_ctr` values for 3 groups:
    # Group 0 - valid, all diffs = 1
    # Group 1 - invalid, has a jump of 5
    # Group 2 - valid, wraps at -16383
    src_seq_ctr = np.array(
        [
            100,
            101,
            102,
            103,  # Group 0
            200,
            205,
            206,
            207,  # Group 1
            16382,
            16383,
            0,
            1,  # Group 2
        ],
        dtype=np.int32,
    )

    group = np.array(
        [
            0,
            0,
            0,
            0,  # Group 0
            1,
            1,
            1,
            1,  # Group 1
            2,
            2,
            2,
            2,  # Group 2
        ]
    )

    grouped_data = xr.Dataset(
        data_vars={"src_seq_ctr": ("epoch", src_seq_ctr)},
        coords={"epoch": epoch, "group": ("epoch", group)},
    )

    return grouped_data


def test_get_pkt_counter(xarray_data):
    """Tests the get_pkt_counter function."""
    status_values = xarray_data["mag_status"].values
    pkt_counter = get_pkt_counter(status_values)
    assert np.array_equal(pkt_counter, np.array([0, 1, 2, 3, 0, 1, 2, 3, 0]))


def test_calculate_time(xarray_data):
    """Tests calculate_time function."""
    time = calculate_time(
        xarray_data["mag_acq_tm_coarse"], xarray_data["mag_acq_tm_fine"]
    )

    assert np.all(
        time
        == xarray_data["mag_acq_tm_coarse"] + xarray_data["mag_acq_tm_fine"] / 65535.0
    )


def test_filter_valid_groups(grouped_data):
    """Tests filter_valid_groups function."""

    filtered_data = filter_valid_groups(grouped_data)

    assert np.all(np.unique(filtered_data["group"]) == np.array([0, 2]))


def test_find_groups(xarray_data):
    """Tests the find_groups function."""
    grouped_data = find_groups(xarray_data)

    assert np.all(np.unique(grouped_data["group"]) == np.array([1, 2]))


def test_get_status_data(xarray_data, mag_test_data):
    """Tests the get_status_data function."""

    status_data = get_status_data(
        xarray_data["mag_status"].values[0:4], np.array([0, 1, 2, 3])
    )
    index = mag_test_data["PRI_COARSETM"] == 461971382
    matching_row = mag_test_data[index]

    for key in status_data.keys():
        assert status_data[key] == matching_row[key.upper()].values[0]


def test_get_time(xarray_data):
    """Tests the get_time function."""
    grouped_data = find_groups(xarray_data)
    time_data = get_time(grouped_data, 1, np.array([0, 1, 2, 3]))
    assert time_data == {
        "pri_coarsetm": 461971382,
        "pri_fintm": 1502,
        "sec_coarsetm": 461971382,
        "sec_fintm": 1505,
    }


def test_get_bytes():
    """Tests the get_bytes function."""

    test_cases = [
        5797207,
        5750698,
        15921110,
        2342918,
        15797207,
        5750697,
        15921110,
        2342918,
    ]

    for val in test_cases:
        extracted = get_bytes(val)

        # Reassemble 24-bit integer from three individual bytes.
        reconstructed_value = (extracted[0] << 16) | (extracted[1] << 8) | extracted[2]

        assert reconstructed_value == val


def test_parse_packet(xarray_data, mag_test_data):
    """Tests the parse_packet function."""
    parsed_packets = parse_packet(xarray_data)

    for packet in parsed_packets:
        index = packet["pri_coarsetm"] == mag_test_data["PRI_COARSETM"]
        matching_rows = mag_test_data[index]

        for key in packet.keys():
            assert packet[key] == matching_rows[key.upper()].values[0]
