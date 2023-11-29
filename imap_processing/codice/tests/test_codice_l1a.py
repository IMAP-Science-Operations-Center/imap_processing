"""Tests the L1a processing for decommutated CoDICE data"""

from pathlib import Path

import pytest

from imap_processing import imap_module_directory
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.codice_l1a import codice_l1a


@pytest.fixture(scope="session")
def l0_test_data():
    """Decom some packets to be used for testing"""
    packet_file = Path(
        f"{imap_module_directory}/codice/tests/data/"
        f"raw_ccsds_20230822_122700Z_idle.bin"
    )
    packets = decom_packets(packet_file)

    return packets


def test_codice_l1a(l0_test_data):
    """Tests the ``codice_l1a`` function and ensured that a proper CDF file
    was created"""

    cdf_filename = codice_l1a(l0_test_data)

    assert Path(cdf_filename).name == "imap_codice_l1a_hk_20100101_v01.cdf"

    # Remove the test data
    Path(cdf_filename).unlink()
