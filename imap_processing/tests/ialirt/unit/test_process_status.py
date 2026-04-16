"""Tests for the process_status module."""

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_status import process_status
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def postlaunch_packet_path():
    """Returns the paths to the binary packets."""
    directory = imap_module_directory / "tests" / "ialirt" / "data" / "l0"
    filenames = [
        "iois_1_packets_2026_090_05_03_05",
        "iois_1_packets_2026_090_05_04_06",
        "iois_1_packets_2026_090_05_05_07",
        "iois_1_packets_2026_090_05_06_08",
        "iois_1_packets_2026_090_05_07_09",
    ]
    return tuple(directory / fname for fname in filenames)


@pytest.fixture
def postlaunch_xarray_data(postlaunch_packet_path, sc_packet_path):
    """Create xarray data for multiple packets."""
    apid = 478
    _, xtce_ialirt_path = sc_packet_path

    xarray_data = tuple(
        packet_file_to_datasets(packet, xtce_ialirt_path, use_derived_value=False)[apid]
        for packet in postlaunch_packet_path
    )

    merged_xarray_data = xr.concat(xarray_data, dim="epoch")
    return merged_xarray_data


@pytest.mark.external_test_data
def test_process_status(postlaunch_xarray_data):
    """Test the process_status function."""

    status_data = process_status(postlaunch_xarray_data)

    for i in range(len(status_data)):
        assert status_data[i]["sc_swapi_status"] == 1
        assert status_data[i]["sc_mag_status"] == 1
        assert status_data[i]["sc_hit_status"] == 1
        assert status_data[i]["sc_codice_status"] == 1
        assert status_data[i]["sc_lo_status"] == 1
        assert status_data[i]["sc_autonomy_status"] == 1
