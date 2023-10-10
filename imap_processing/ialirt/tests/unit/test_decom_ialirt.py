import sys
from pathlib import Path

import pytest

from imap_processing.ialirt.l0.decom_ialirt import decom_packets, generate_xarray


@pytest.fixture(scope="session")
def packet_path():
    """Returns the raw packet directory."""
    return Path(sys.modules[__name__.split(
        '.')[0]].__file__).parent / 'ialirt' / 'tests' / 'test_data' / 'l0' / \
        'IALiRT Raw Packet Telemetry.txt'


@pytest.fixture(scope="session")
def xtce_ialirt_path():
    """Returns the xtce auxiliary directory."""
    return Path(sys.modules[__name__.split(
        '.')[0]].__file__).parent / 'ialirt' / 'packet_definitions' \
        / "ialirt.xml"


@pytest.fixture(scope="session")
def decom_data(packet_path, xtce_ialirt_path):
    """Data for decom_ultra"""
    data_packet_list = generate_xarray(packet_path, xtce_ialirt_path)
    return data_packet_list


@pytest.fixture(scope="session")
def decom_test_data(packet_path, xtce_ialirt_path):
    """Read test data from file"""
    data_packet_list = decom_packets(packet_path, xtce_ialirt_path)
    return data_packet_list


def test_length(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 32
    assert len(decom_test_data) == total_packets


def test_enumerated(decom_test_data):
    """Test if enumerated values derived correctly"""

    for packet in decom_test_data:

        assert packet.data["SC_SWAPI_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_MAG_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_HIT_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_CODICE_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_LO_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_HI_45_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_HI_90_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_ULTRA_45_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_ULTRA_90_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_SWE_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_IDEX_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_GLOWS_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SC_SPINPERIODVALID"].derived_value == 'INVALID'
        assert packet.data["SC_SPINPHASEVALID"].derived_value == 'INVALID'
        assert packet.data["SC_ATTITUDE"].derived_value == 'SUNSENSOR'
        assert packet.data["SC_CATBEDHEATERFLAG"].derived_value == 'ON'
        assert packet.data["SC_AUTONOMY"].derived_value == 'OPERATIONAL'
        assert packet.data["HIT_STATUS"].derived_value == 'OFF-NOMINAL'
        assert packet.data["SWE_NOM_FLAG"].derived_value == 'OFF-NOMINAL'
        assert packet.data["SWE_OPS_FLAG"].derived_value == 'NON-HVSCI'


def test_decom_instruments(decom_data, decom_test_data):
    """This function checks that all instrument parameters are accounted for."""

    instrument_lengths = {}
    instruments = ['SC', 'HIT', 'MAG', 'COD_LO', 'COD_HI', 'SWE', 'SWAPI']

    # Assert that all parameters are the same
    # between packet data and generated xarray
    for instrument in instruments:
        instrument_list = list(decom_data[instrument].coords)
        instrument_list.extend(list(decom_data[instrument].data_vars))

        instrument_lengths[instrument] = len(instrument_list)

        assert all(param in list(decom_test_data[0].data.keys())
                   for param in instrument_list)

    # Assert that the length of parameters is the same
    # between packet data and generated xarray
    assert sum(instrument_lengths.values()) == len(list(decom_test_data[0].data.keys()))
