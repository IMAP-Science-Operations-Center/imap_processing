import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.parse_mag import (
    extract_magnetic_vectors,
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
    """Creates grouped data for tests."""
    epoch = np.arange(12)

    # Example `src_seq_ctr` values for 3 groups:
    # Group 0 - valid, all diffs = 1
    # Group 1 - invalid, has a jump of 5
    # Group 2 - valid, wraps at -16383
    src_seq_ctr = np.concatenate(
        [
            np.arange(100, 104),
            np.array([200, 205, 206, 207]),
            np.array([16382, 16383, 0, 1]),
        ],
        dtype=np.int32,
    )
    mag_acq_tm_coarse = np.repeat(
        np.array([461971382, 461971386, 461971390], dtype=np.uint32), repeats=4
    )

    mag_acq_tm_fine = np.array(
        [1502, 1502, 1505, 1505, 1500, 1500, 1503, 1503, 1497, 1497, 1491, 1491]
    )

    group = np.tile(np.arange(3), 4).reshape(4, 3).T.ravel()

    grouped_data = xr.Dataset(
        data_vars={
            "src_seq_ctr": ("epoch", src_seq_ctr),
            "mag_acq_tm_coarse": ("epoch", mag_acq_tm_coarse),
            "mag_acq_tm_fine": ("epoch", mag_acq_tm_fine),
        },
        coords={"epoch": epoch, "group": ("epoch", group)},
    )

    return grouped_data


def test_get_pkt_counter(xarray_data):
    """Tests the get_pkt_counter function."""
    status_values = xarray_data["mag_status"].values
    pkt_counter = get_pkt_counter(status_values)
    assert np.array_equal(pkt_counter, np.array([0, 1, 2, 3, 0, 1, 2, 3, 0]))


def test_get_status_data(xarray_data, mag_test_data):
    """Tests the get_status_data function."""

    status_data = get_status_data(
        xarray_data["mag_status"].values[0:4], np.array([0, 1, 2, 3])
    )
    index = mag_test_data["PRI_COARSETM"] == 461971382
    matching_row = mag_test_data[index]

    for key in status_data.keys():
        assert status_data[key] == matching_row[key.upper()].values[0]


def test_get_time(grouped_data):
    """Tests the get_time function."""
    time_data = get_time(grouped_data, 1, np.array([0, 1, 2, 3]))
    assert time_data == {
        "pri_coarsetm": 461971386,
        "pri_fintm": 1500,
        "sec_coarsetm": 461971386,
        "sec_fintm": 1503,
    }


def test_extract_magnetic_vectors():
    """Tests the extract_magnetic_vectors function."""
    science_values = xr.DataArray(
        data=np.array([15797207, 5750698, 15921110, 2342918], dtype=np.uint32)
    )

    vectors = extract_magnetic_vectors(science_values)

    assert vectors == {
        "pri_x": 61707,
        "pri_y": 55127,
        "pri_z": 49066,
        "sec_x": 62191,
        "sec_y": 54819,
        "sec_z": 49158,
    }


def test_parse_packet(xarray_data, mag_test_data):
    """Tests the parse_packet function."""
    parsed_packets = parse_packet(xarray_data)

    for packet in parsed_packets:
        index = packet["pri_coarsetm"] == mag_test_data["PRI_COARSETM"]
        matching_rows = mag_test_data[index]

        for key in packet.keys():
            assert packet[key] == matching_rows[key.upper()].values[0]
