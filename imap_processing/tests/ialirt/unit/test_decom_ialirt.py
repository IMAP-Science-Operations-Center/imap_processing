import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets
from imap_processing.utils import packet_file_to_datasets

IALIRT_PACKET_LENGTH = 1464


@pytest.fixture(scope="session")
def xtce_ialirt_path():
    """Returns the xtce auxiliary directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"


@pytest.fixture()
def binary_packet_path(tmp_path):
    """
    Creates a binary file from the text packet data, which is more representative
    of the actual operational environment. The binary file is deleted after the
    test session.
    """
    packet_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "IALiRT Raw Packet Telemetry.txt"
    )

    binary_file_path = tmp_path / "file.ccsds"

    with open(packet_path) as text_file, open(binary_file_path, "wb") as binary_file:
        for line in text_file:
            if not line.startswith("#"):
                # Split the line by semicolons
                # Discard the first value since it is only a counter
                hex_values = line.strip().split(";")[1:]
                # Convert hex to binary
                binary_data = bytearray.fromhex("".join(hex_values))
                binary_file.write(binary_data)
                assert len(binary_data) * 8 == IALIRT_PACKET_LENGTH

    return binary_file_path


@pytest.fixture()
def decom_packets_data(binary_packet_path, xtce_ialirt_path):
    """Read packet data from file using decom_packets"""
    data_packet_list = decom_packets(binary_packet_path, xtce_ialirt_path)
    return data_packet_list


def test_length(decom_packets_data):
    """Test if total packets in data file is correct"""
    total_packets = 32
    assert len(decom_packets_data) == total_packets


def test_enumerated(decom_packets_data):
    """Test if enumerated values derived correctly"""

    for packet in decom_packets_data:
        assert packet["SC_SWAPI_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_MAG_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_HIT_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_CODICE_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_LO_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_HI_45_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_HI_90_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_ULTRA_45_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_ULTRA_90_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_SWE_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_IDEX_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_GLOWS_STATUS"] == "NOT_OPERATIONAL"
        assert packet["SC_SPINPERIODVALID"] == "INVALID"
        assert packet["SC_SPINPHASEVALID"] == "INVALID"
        assert packet["SC_ATTITUDE"] == "SUNSENSOR"
        assert packet["SC_CATBEDHEATERFLAG"] == "ON"
        assert packet["SC_AUTONOMY"] == "OPERATIONAL"
        assert packet["HIT_STATUS"] == "OFF-NOMINAL"
        assert packet["SWE_NOM_FLAG"] == "OFF-NOMINAL"
        assert packet["SWE_OPS_FLAG"] == "NON-HVSCI"


def test_generate_xarray(binary_packet_path, xtce_ialirt_path, decom_packets_data):
    """This function checks that all instrument parameters are correct length."""

    apid = 478
    xarray_data = packet_file_to_datasets(binary_packet_path, xtce_ialirt_path)[apid]

    for key in xarray_data.keys():
        assert len(xarray_data[key]) == 32
