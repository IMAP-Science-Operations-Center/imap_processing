import pytest
from pathlib import Path
from imap_processing import decom, packet_definition_directory
from imap_processing.ultra import decom_ultra
import pandas as pd
import ast
import numpy as np


@pytest.fixture(scope="function")
def decom_test_data(ccsds_path, xtce_aux_path):
    """Read test data from file"""
    data_packet_list = decom.decom_packets(ccsds_path, xtce_aux_path)
    return data_packet_list


def test_total_packets_in_data_file(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 22
    for packet in decom_test_data:
        print('hi')
    assert len(decom_test_data) == total_packets


def test_enumerated_value_880(decom_test_data):
    """Test if enumerated value is derived correctly"""

    # CEM Nominal status bit:
    #     '1' -- nominal,
    #     '0' -- not nomimal
    parameter_name = "SPINPERIODVALID"

    index_of_first_880 = None  # We'll store the index here if we find it

    for index, packet in enumerate(decom_test_data):
        if packet.header['PKT_APID'].derived_value == 880:
            index_of_first_880 = index
            break

    first_packet_data = decom_test_data[index_of_first_880].data

    if first_packet_data[f"{parameter_name}"].raw_value == 1:
        assert first_packet_data[f"{parameter_name}"].derived_value == "VALID"
    if first_packet_data[f"{parameter_name}"].raw_value == 0:
        assert first_packet_data[f"{parameter_name}"].derived_value == "INVALID"



