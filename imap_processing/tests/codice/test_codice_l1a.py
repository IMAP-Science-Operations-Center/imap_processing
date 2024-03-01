"""Tests the L1a processing for decommutated CoDICE data"""

from pathlib import Path

import pytest
import space_packet_parser

from imap_processing import imap_module_directory
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.codice_l1a import codice_l1a


@pytest.fixture(scope="session")
def l0_test_data() -> list:
    """Decom some packets to be used for testing

    Returns
    -------
    packets : list[space_packet_parser.parser.Packet]
        A list of decommutated packets for testing
    """

    packet_file = Path(
        f"{imap_module_directory}/tests/codice/data/"
        f"raw_ccsds_20230822_122700Z_idle.bin"
    )
    packets = decom_packets(packet_file)

    return packets


def test_codice_l1a(l0_test_data: list[space_packet_parser.parser.Packet]) -> str:
    """Tests the ``codice_l1a`` function and ensured that a proper CDF file
    was created

    Parameters
    ----------
    l0_test_data : list[space_packet_parser.parser.Packet]
        A list of packets to process
    tmp_path : pathlib.PosixPath
        pytest fixture used to provide a temporary directory during testing
    """

    cdf_filename = codice_l1a(l0_test_data)

    assert Path(cdf_filename).name == "imap_codice_l1a_hk_20210101_20210102_v01-01.cdf"
