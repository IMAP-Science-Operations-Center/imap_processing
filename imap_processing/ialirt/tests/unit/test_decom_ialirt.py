import pytest

from imap_processing.ialirt.l0.decom_ialirt import decom_packets
from imap_processing.ialirt.tests.unit.data_path_fixtures import \
    packet_path, xtce_ialirt_path


@pytest.fixture()
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

        assert packet.data["SWAPI_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["MAG_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["HIT_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["CODICE_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["LO_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["HI_45_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["HI_90_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["ULTRA_45_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["ULTRA_90_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SWE_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["IDEX_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["GLOWS_STATUS"].derived_value == 'NOT_OPERATIONAL'
        assert packet.data["SPINPERIODVALID"].derived_value == 'INVALID'
        assert packet.data["SPINPHASEVALID"].derived_value == 'INVALID'
        assert packet.data["ATTITUDE"].derived_value == 'SUNSENSOR'
        assert packet.data["CATBEDHEATERFLAG"].derived_value == 'ON'
        assert packet.data["AUTONOMY"].derived_value == 'OPERATIONAL'
