import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets


@pytest.fixture(scope="session")
def decom_nhk_data():
    """Read test housekeeping data from test folder"""
    test_folder_path = "tests/hi/l0_test_data"
    packet_file = imap_module_directory / f"{test_folder_path}/20231030_H45_APP_NHK.bin"
    packet_def_file = imap_module_directory / "hi/packet_definitions/H45_APP_NHK.xml"
    return decom_packets(packet_file, packet_def_file)


@pytest.fixture(scope="session")
def nhk_validation_data():
    """Read in validation data from the CSV file"""
    test_data_path = imap_module_directory / "tests/hi/l0_validation_data"
    raw_validation_data = pd.read_csv(
        test_data_path / "20231030_H45_APP_NHK.csv",
        index_col="CCSDS_MET",
    )
    return raw_validation_data


@pytest.fixture(scope="session")
def decom_sci_de_data():
    """Read science direct event data from test folder"""
    test_folder_path = "tests/hi/l0_test_data"
    packet_file = imap_module_directory / f"{test_folder_path}/20231030_H45_SCI_DE.bin"
    packet_def_file = imap_module_directory / "hi/packet_definitions/H45_SCI_DE.xml"
    return decom_packets(packet_file, packet_def_file)


@pytest.fixture(scope="session")
def decom_sci_cnt_data():
    """Read science count data from test folder"""
    test_folder_path = "tests/hi/l0_test_data"
    packet_file = imap_module_directory / f"{test_folder_path}/20231030_H45_SCI_CNT.bin"
    packet_def_file = imap_module_directory / "hi/packet_definitions/H45_SCI_CNT.xml"
    return decom_packets(packet_file, packet_def_file)


def test_app_nhk_decom(decom_nhk_data):
    """Test housekeeping data"""
    expected_number_of_packets = 100
    assert len(decom_nhk_data) == expected_number_of_packets
    # TODO: extend test more and check with validation data
    # Validation data seems off. TODO: check with IMAP-Hi team


def test_sci_de_decom(decom_sci_de_data):
    """Test science direct event data"""
    expected_number_of_packets = 100
    assert len(decom_sci_de_data) == expected_number_of_packets


def test_sci_cnt_decom(decom_sci_cnt_data):
    "Test science count data"
    expected_number_of_packets = 100
    assert len(decom_sci_cnt_data) == expected_number_of_packets
