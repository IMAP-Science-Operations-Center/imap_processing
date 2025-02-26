import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.utils import packet_file_to_datasets
from imap_processing.ialirt.l0.parse_mag import get_pkt_counter, get_status_data, parse_packet, find_groups


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
        "461971390-411.bin"
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
        / "sample decoded i-alirt data.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture()
def xarray_data(binary_packet_path, xtce_mag_path):
    """Create xarray data."""
    apid = 1001

    xarray_data = tuple(
        packet_file_to_datasets(packet, xtce_mag_path, use_derived_value=False)[apid]
        for packet in binary_packet_path
    )

    merged_xarray_data = xr.concat(xarray_data, dim="epoch")
    return merged_xarray_data


def test_get_pkt_counter(xarray_data):
    """Tests the get_pkt_counter function."""
    for i, ds in enumerate(xarray_data):
        expected = i % 4
        pkt_counter = get_pkt_counter(int(ds["mag_status"].values[0]))
        assert pkt_counter == expected, f"Expected {expected}, got {pkt_counter}"


def test_get_science_data(xarray_data, mag_test_data):
    """Tests the get_science_data function."""

    science_data_0 = get_status_data(int(xarray_data["mag_status"][0].values), 0)
    science_data_1 = get_status_data(int(xarray_data["mag_status"][1].values), 1)
    science_data_2 = get_status_data(int(xarray_data["mag_status"][2].values), 2)
    science_data_3 = get_status_data(int(xarray_data["mag_status"][3].values), 3)

    print('hi')

def test_find_groups(xarray_data):
    """Tests the find_groups function."""
    grouped_data = find_groups(xarray_data)

    assert len(np.unique(grouped_data["mag_acq_tm_coarse"])) == len(
        np.unique(grouped_data["group"]))


def test_parse_packet(xarray_data):
    """Tests the parse_packet function."""
    parsed_packet = parse_packet(xarray_data)

