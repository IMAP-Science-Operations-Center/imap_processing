"""Tests the decommutation process for CoDICE CCSDS Packets"""

from pathlib import Path

import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = Path(f"{imap_module_directory}/codice/tests/housekeeping_data.bin")
    xtce_document = Path(
        f"{imap_module_directory}/codice/packet_definitions/P_COD_NHK.xml"
    )
    data_packet_list = decom_packets(packet_file, xtce_document)
    data_packet_list = [
        packet
        for packet in data_packet_list
        if packet.header["PKT_APID"].raw_value == 1136
    ]

    return data_packet_list

@pytest.fixture(scope="session")
def validataion_data():
    """Read in validation data from the CSV file

    Returns
    -------
    validatation_data : pandas Dataframe?
        The validation data read from the CSV, cleaned up and ready to compare
        the decommutated packet with
    """

    # Read in the CSV file (perhaps to a pandas dataframe?)

    # Remove the timestamp column and data

    # Return the data

    pass


def test_housekeeping_data(decom_test_data, validataion_data):
    """Compare the decommutated housekeeping data and compare it to the
    validataion data to make sure they are the same.

    Parameters
    ----------
    decom_test_data : dict
        The decommuated housekeeping packet data
    validatation_data : pandas Dataframe?
        The validation data to compare against
    """

    # May need to apply Analog Conversions to specific mnemonics first?

    # Compare keywords and values, similar to how SWE does it (i.e. test_swe_housekeeping.py)

    pass


def test_total_packets_in_data_file(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 329
    assert len(decom_test_data) == total_packets


def test_ways_to_get_data(decom_test_data):
    """Test if data can be retrieved using different ways"""
    # First way to get data
    data_value_using_key = decom_test_data[0].data

    # Second way to get data
    data_value_using_list = decom_test_data[0][1]
    # Check if data is same
    assert data_value_using_key == data_value_using_list
