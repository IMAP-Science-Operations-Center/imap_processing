import pytest

from imap_processing.idex.IDEXPacketParser import IDEXPacketParser


@pytest.fixture(scope="session")
def decom_test_data():
    return IDEXPacketParser("imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts")

def test_idex_decom_length(decom_test_data):
    assert len(decom_test_data.data) == 6

def test_idex_decom_event_num(decom_test_data):
    for var in decom_test_data.data:
        assert len(decom_test_data.data[var]) == 19
