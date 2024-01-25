import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.l0 import decom_swe


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    test_folder_path = imap_module_directory / "tests/swe/l0_data"
    packet_1_file = test_folder_path / "20230927100425_SWE_CEM_RAW_packet.bin"
    packet_2_file = test_folder_path / "20230927100426_SWE_CEM_RAW_packet.bin"
    first_data = decom_swe.decom_packets(packet_1_file)
    second_data = decom_swe.decom_packets(packet_2_file)

    return first_data + second_data


def test_number_of_packets(decom_test_data):
    """This test and validate number of packets."""
    expected_number_of_packets = 2
    assert len(decom_test_data) == expected_number_of_packets


def test_swe_raw_cem_data(decom_test_data):
    """This test and validate raw data of SWE raw CEM data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    raw_validation_data = pd.read_csv(
        test_data_path / "data_raw.SWE_CEM_RAW_20230927_094839.csv",
        index_col="SHCOARSE",
    )
    first_data = decom_test_data[0]
    validation_data = raw_validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # compare raw values of housekeeping data
    for key, value in first_data.data.items():
        # check if the data is the same, for SHCOARSE we need the name of the column.
        # This is done because pandas removed it from the main columns to make it the
        # index.
        assert value.raw_value == (
            validation_data[key] if key != "SHCOARSE" else validation_data.name
        )
