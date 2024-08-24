import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.utils.swe_utils import SWEAPID
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = (
        imap_module_directory / "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    )
    xtce_document = (
        imap_module_directory / "swe/packet_definitions/swe_packet_definition.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file, xtce_document, use_derived_value=False
    )
    return datasets_by_apid[SWEAPID.SWE_SCIENCE]


@pytest.fixture(scope="session")
def decom_test_data_derived():
    """Read test data from file"""
    packet_file = (
        imap_module_directory / "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    )
    xtce_document = (
        imap_module_directory / "swe/packet_definitions/swe_packet_definition.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file, xtce_document, use_derived_value=True
    )
    return datasets_by_apid[SWEAPID.SWE_SCIENCE]
