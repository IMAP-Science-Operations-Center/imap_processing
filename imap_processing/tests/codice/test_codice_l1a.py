"""Tests the L1a processing for decommutated CoDICE data"""

from pathlib import Path

import pytest

from imap_processing import imap_module_directory
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.codice_l1a import process_codice_l1a

TEST_DATA = [
    (
        Path(
            f"{imap_module_directory}/tests/codice/data/raw_ccsds_20230822_122700Z_idle.bin"
        ),
        Path(f"{imap_module_directory}/codice/packet_definitions/P_COD_NHK.xml"),
        "imap_codice_l1a_hskp_20100101_v001.cdf",
    ),
    (
        Path(f"{imap_module_directory}/tests/codice/data/lo_fsw_view_5_ccsds.bin"),
        Path(
            f"{imap_module_directory}/codice/packet_definitions/P_COD_LO_SW_SPECIES_COUNTS.xml"
        ),
        "imap_codice_l1a_lo-sw-species-counts_20240319_v001.cdf",
    ),
]


@pytest.mark.parametrize(("test_file", "xtce_document", "expected_filename"), TEST_DATA)
def test_codice_l1a(test_file: Path, xtce_document: Path, expected_filename: str):
    """Tests the ``process_codice_l1a`` function and ensure that a proper CDF
    files are created.

    Parameters
    ----------
    test_file : Path
        The file containing test data
    xtce_document: Path
        Path to the XTCE document file.
    expected_filename : str
        The filename of the generated CDF file
    """

    packets = decom_packets(test_file, xtce_document)
    cdf_filename = process_codice_l1a(packets)
    assert cdf_filename.name == expected_filename
