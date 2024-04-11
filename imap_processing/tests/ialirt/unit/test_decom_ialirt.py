import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets
from imap_processing.ialirt.l0.decom_ialirt import generate_xarray

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
        assert packet.data["SC_SWAPI_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_MAG_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_HIT_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_CODICE_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_LO_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_HI_45_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_HI_90_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_ULTRA_45_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_ULTRA_90_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_SWE_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_IDEX_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_GLOWS_STATUS"].derived_value == "NOT_OPERATIONAL"
        assert packet.data["SC_SPINPERIODVALID"].derived_value == "INVALID"
        assert packet.data["SC_SPINPHASEVALID"].derived_value == "INVALID"
        assert packet.data["SC_ATTITUDE"].derived_value == "SUNSENSOR"
        assert packet.data["SC_CATBEDHEATERFLAG"].derived_value == "ON"
        assert packet.data["SC_AUTONOMY"].derived_value == "OPERATIONAL"
        assert packet.data["HIT_STATUS"].derived_value == "OFF-NOMINAL"
        assert packet.data["SWE_NOM_FLAG"].derived_value == "OFF-NOMINAL"
        assert packet.data["SWE_OPS_FLAG"].derived_value == "NON-HVSCI"


def test_generate_xarray(binary_packet_path, xtce_ialirt_path, decom_packets_data):
    """This function checks that all instrument parameters are accounted for."""

    instrument_lengths = {}
    instruments = ["SC", "HIT", "MAG", "COD_LO", "COD_HI", "SWE", "SWAPI"]

    xarray_data = generate_xarray(binary_packet_path, xtce_ialirt_path)

    # Assert that all parameters are the same
    # between packet data and generated xarray
    for instrument in instruments:
        instrument_list = list(xarray_data[instrument].coords)
        instrument_list.extend(list(xarray_data[instrument].data_vars))

        # Create a dictionary of the number of parameters for each instrument
        instrument_lengths[instrument] = len(instrument_list)

        packet_data_keys_set = set(decom_packets_data[0].data.keys())
        instrument_list_set = set(instrument_list)

        # Assert that the instrument parameters from the xarray are
        # a subset of the packet data
        assert instrument_list_set.issubset(packet_data_keys_set) is True

    # Assert that the length of parameters is the same
    # between packet data and generated xarray
    assert sum(instrument_lengths.values()) == len(
        list(decom_packets_data[0].data.keys())
    )

    # Check that all the dimensions are the correct length
    # example:len(xarray_data['HIT'].coords['HIT_SC_TICK'].values) == 32
    for instrument in instruments:
        for dimension_name in xarray_data[instrument].dims:
            assert len(xarray_data[instrument].coords[dimension_name].values) == 32


def test_unexpected_key(binary_packet_path, xtce_ialirt_path, decom_packets_data):
    """
    This function tests that a ValueError is raised when an unexpected
    key is encountered.
    """

    time_keys = {
        "UNEXPECTED": "SC_SCLK_SEC",
        "HIT": "HIT_SC_TICK",
        "MAG": "MAG_ACQ",
        "COD_LO": "COD_LO_ACQ",
        "COD_HI": "COD_HI_ACQ",
        "SWE": "SWE_ACQ_SEC",
        "SWAPI": "SWAPI_ACQ",
    }

    with pytest.raises(ValueError, match=r"Unexpected key '.*' found in packet data."):
        generate_xarray(binary_packet_path, xtce_ialirt_path, time_keys=time_keys)
