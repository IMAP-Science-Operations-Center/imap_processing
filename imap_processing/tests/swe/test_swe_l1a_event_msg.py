import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.l0 import decom_swe


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    test_folder_path = "tests/swe/l0_data"
    event_file_list = list(
        imap_module_directory.glob(f"{test_folder_path}/*SWE_EVTMSG_packet.bin")
    )
    event_data = []
    for packet_file in event_file_list:
        event_data.append(decom_swe.decom_packets(packet_file)[0])

    return event_data


def test_number_of_packets(decom_test_data):
    """This test and validate number of packets."""
    expected_number_of_packets = 15
    assert len(decom_test_data) == expected_number_of_packets


def test_swe_event_msg_data(decom_test_data):
    """This test and validate raw data of SWE event message data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"

    raw_validation_data = pd.read_csv(
        test_data_path / "idle_export_raw.SWE_EVTMSG_20231004_140149.csv",
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
