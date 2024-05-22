import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.l0 import decom_swe


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = (
        imap_module_directory / "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    )
    return decom_swe.decom_packets(packet_file)
