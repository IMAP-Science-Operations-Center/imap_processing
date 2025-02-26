import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.utils import packet_file_to_datasets
from imap_processing.ialirt.l0.parse_mag import get_pkt_counter, get_science_data


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
    return tuple(
        packet_file_to_datasets(packet, xtce_mag_path, use_derived_value=False)[apid]
        for packet in binary_packet_path
    )


def test_get_pkt_counter(xarray_data):
    """Tests the get_pkt_counter function."""
    for i, ds in enumerate(xarray_data):
        expected = i % 4
        pkt_counter = get_pkt_counter(int(ds["mag_status"].values[0]))
        assert pkt_counter == expected, f"Expected {expected}, got {pkt_counter}"


def test_get_science_data(xarray_data):
    """Tests the get_science_data function."""

    xarray_data_0, xarray_data_1, xarray_data_2, xarray_data_3,\
        xarray_data_4, xarray_data_5, xarray_data_6, xarray_data_7= xarray_data

    science_data_0 = get_science_data(int(xarray_data_0["mag_data"].values[0]), 0)
    #science_data_1 = get_science_data(int(xarray_data_1["mag_data"].values[0]), 1)
    science_data_2 = get_science_data(int(xarray_data_2["mag_status"].values[0]), 2)
    #science_data_3 = get_science_data(int(xarray_data_3["mag_data"].values[0]), 3)

    print('hi')

