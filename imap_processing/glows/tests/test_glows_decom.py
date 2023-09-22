import pytest

from imap_processing.glows.l0 import glows_decom


@pytest.fixture()
def decom_test_data():
    """Read test data from file"""
    packet_path = "imap_l0_sci_glows_20230920_v00.pcts"
    data_packet_list = glows_decom.decom_packets(packet_path)
    return data_packet_list


def test_glows_hist_decom(decom_test_data):
    expected_len = 13011

    assert len(decom_test_data) == expected_len
