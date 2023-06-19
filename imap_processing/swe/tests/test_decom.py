from pathlib import Path

import pytest

from imap_processing import decom


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = Path(
        "imap_processing/swe/tests/science_block_20221116_163611Z_idle.bin"
    )
    xtce_document = Path("imap_processing/swe/swe_packet_definition.xml")
    data_packet_list = decom.decom_packets(packet_file, xtce_document)
    return data_packet_list


def test_total_packets_in_data_file(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 23
    assert len(decom_test_data) == total_packets


def test_ccsds_header(decom_test_data):
    """Test if packet header contains default CCSDS header

    These are the field required in CCSDS header:
        'VERSION', 'TYPE', 'SEC_HDR_FLG', 'PKT_APID', 'SEG_FLGS', 'SRC_SEQ_CTR', 'PKT_LEN'
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
    # is using list method. Eg. ccsds_header = decom_test_data[0][0]. Each packet's 0th index
    # has header data and index 1 has data.

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


def test_enumerated_value(decom_test_data):
    """Test if enumerated value is derived correctly"""

    # CEM Nominal status bit:
    #     '1' -- nominal,
    #     '0' -- not nomimal
    parameter_name = "CEM_NOMINAL_ONLY"
    first_packet_data = decom_test_data[0].data

    if first_packet_data[f"{parameter_name}"].raw_value == 1:
        assert first_packet_data[f"{parameter_name}"].derived_value == "NOMINAL"
    if first_packet_data[f"{parameter_name}"].raw_value == 0:
        assert first_packet_data[f"{parameter_name}"].derived_value == "NOT_NOMINAL"
