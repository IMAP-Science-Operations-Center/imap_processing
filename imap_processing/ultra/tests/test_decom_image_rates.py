import pytest
from pathlib import Path
import logging

from imap_processing.ultra import decom_ultra


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = Path('../test_data/Ultra45_EM_SwRI_Cal_Run7_ThetaScan_20220530T225054.CCSDS')
    data_packet_list = decom_ultra.decom_packets(packet_file, "P_U45_IMAGE_RATES.xml")
    return data_packet_list

def test_total_packets_in_data_file(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 23
    assert len(decom_test_data) == total_packets


def test_ccsds_header(decom_test_data):
    """Test if packet header contains default CCSDS header

    These are the field required in CCSDS header:
        'VERSION', 'TYPE', 'SEC_HDR_FLG', 'PKT_APID',
        'SEG_FLGS', 'SRC_SEQ_CTR', 'PKT_LEN'
    """
    # Required CCSDS header fields
    ccsds_header_keys = [
        "VERSION",
        "TYPE",
        "SEC_HDR_FLG",
        "PKT_APID",
        "SEG_FLGS",
        "SRC_SEQ_CTR",
        "PKT_LEN",
    ]

    # decom_test_data[0].header is one way to get the header data. Another way to get it
    # is using list method. Eg. ccsds_header = decom_test_data[0][0].
    # Each packet's 0th index has header data and index 1 has data.

    # First way to get header data
    ccsds_header = decom_test_data[0].header
    assert all(key in ccsds_header.keys() for key in ccsds_header_keys)

    # Second way to get header data
    ccsds_header = decom_test_data[0][0]
    assert all(key in ccsds_header.keys() for key in ccsds_header_keys)


def test_ways_to_get_data(decom_test_data):
    """Test if data can be retrieved using different ways"""
    # First way to get data
    data_value_using_key = decom_test_data[0].data

    # Second way to get data
    data_value_using_list = decom_test_data[0][1]
    # Check if data is same
    assert data_value_using_key == data_value_using_list


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


def test_enumerated_value_881(decom_test_data):
    """Test if enumerated value is derived correctly"""

    logging.basicConfig(level='DEBUG')

    index_of_first_881 = None

    for index, packet in enumerate(decom_test_data):
        if packet.header['PKT_APID'].derived_value == 881:
            index_of_first_881 = index
            print('hi')

    first_packet_data = decom_test_data[index_of_first_881].data

    # Extract raw FASTDATA_00 value from first_packet_data
    fastdata_raw = first_packet_data['FASTDATA_00'].raw_value

    # Do further assertions or validations
    return

