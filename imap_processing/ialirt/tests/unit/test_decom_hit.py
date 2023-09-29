import pytest

from imap_processing.ialirt.l0.decom_ialirt import decom_packets
from imap_processing.ialirt.tests.unit.data_path_fixtures import \
    packet_path, xtce_ialirt_path
from imap_processing.ialirt.l0.decom_hit import decom_hit_packets


@pytest.fixture()
def decom_hit(packet_path, xtce_ialirt_path):
    """Data for decom_ultra"""
    data_packet_list = decom_hit_packets(packet_path,  xtce_ialirt_path)
    return data_packet_list


def test_decom_hit(decom_hit):
    """This function reads validation data and checks that
    decom data matches validation data for auxiliary packet"""

    print('hi')


