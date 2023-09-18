import numpy as np
import pytest

from imap_processing.idex.idex_packet_parser import PacketParser


@pytest.fixture(scope="session")
def decom_test_data():
    return PacketParser("imap_processing/idex/tests/imap_idex_l0_20230725_v01-00.pkts")


def test_idex_decom_length(decom_test_data):
    # Verify that there are 6 data variables in the output
    assert len(decom_test_data.l1_data) == 42


def test_idex_decom_event_num(decom_test_data):
    # Verify that 19 impacts were gathered by the test data
    for var in decom_test_data.l1_data:
        assert len(decom_test_data.l1_data[var]) == 19


def test_idex_tof_high_data(decom_test_data):
    # Verify that a sample of the data is correct
    # impact_14_tof_high_data.txt has been verified correct by the IDEX team
    with open("imap_processing/idex/tests/impact_14_tof_high_data.txt") as f:
        data = np.array([int(line.rstrip("\n")) for line in f])
    assert (decom_test_data.l1_data["TOF_High"][13].data == data).all()
