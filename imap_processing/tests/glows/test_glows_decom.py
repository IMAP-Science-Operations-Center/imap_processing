from collections import namedtuple
from pathlib import Path

import pytest
from space_packet_parser.parser import ParsedDataItem

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.glows.l0 import decom_glows


@pytest.fixture(scope="module")
def decom_test_data():
    """Read test data from file"""
    current_directory = Path(__file__).parent
    packet_path = current_directory / "glows_test_packet_20110921_v01.pkts"
    data_packet_list = decom_glows.decom_packets(packet_path)
    return data_packet_list


def test_glows_decom_count(decom_test_data):
    expected_hist_packet_count = 505
    expected_de_packet_count = 1088

    assert len(decom_test_data) == 2

    # Histogram data
    assert len(decom_test_data[0]) == expected_hist_packet_count

    # Direct events data
    assert len(decom_test_data[1]) == expected_de_packet_count


def test_glows_hist_data(decom_test_data):
    expected_data = {
        "MET": 54232338,
        "STARTID": 0,
        "ENDID": 0,
        "FLAGS": 64,
        "SWVER": 131329,
        "SEC": 54232215,
        "SUBSEC": 0,
        "OFFSETSEC": 120,
        "OFFSETSUBSEC": 0,
        "GLXSEC": 54232214,
        "GLXSUBSEC": 1997263,
        "GLXOFFSEC": 119,
        "GLXOFFSUBSEC": 1998758,
        "SPINS": 7,
        "NBINS": 3600,
        "TEMPAVG": 203,
        "TEMPVAR": 22,
        "HVAVG": 2007,
        "HVVAR": 0,
        "SPAVG": 46875,
        "SPVAR": 0,
        "ELAVG": 6,
        "ELVAR": 0,
        "EVENTS": 95978,
    }
    for key in expected_data.keys():
        assert getattr(decom_test_data[0][0], key) == expected_data[key]


def test_glows_de_data(decom_test_data):
    expected_data = {"MET": 54233694, "SEC": 54232338, "LEN": 1, "SEQ": 0}
    for key in expected_data.keys():
        assert getattr(decom_test_data[1][0], key) == expected_data[key]


def test_bad_header():
    bad_data = {"test": namedtuple("TestData", ["derived_value"])}
    bad_data["test"].derived_value = "test"
    with pytest.raises(KeyError, match="Did not find matching"):
        CcsdsData(bad_data)


def test_header(decom_test_data):
    expected_hist = CcsdsData(
        {
            "VERSION": ParsedDataItem("VERSION", 0, unit=None),
            "TYPE": ParsedDataItem("TYPE", 0, unit=None),
            "SEC_HDR_FLG": ParsedDataItem("SEC_HDR_FLG", 1, unit=None),
            "PKT_APID": ParsedDataItem("PKT_APID", 1480, unit=None),
            "SEQ_FLGS": ParsedDataItem("SEQ_FLGS", 3, unit=None),
            "SRC_SEQ_CTR": ParsedDataItem("SRC_SEQ_CTR", 0, unit=None),
            "PKT_LEN": ParsedDataItem("PKT_LEN", 3663, unit=None),
        }
    )

    assert expected_hist == decom_test_data[0][0].ccsds_header
    expected_de = CcsdsData(
        {
            "VERSION": ParsedDataItem("VERSION", 0, unit=None),
            "TYPE": ParsedDataItem("TYPE", 0, unit=None),
            "SEC_HDR_FLG": ParsedDataItem("SEC_HDR_FLG", 1, unit=None),
            "PKT_APID": ParsedDataItem("PKT_APID", 1481, unit=None),
            "SEQ_FLGS": ParsedDataItem("SEQ_FLGS", 3, unit=None),
            "SRC_SEQ_CTR": ParsedDataItem("SRC_SEQ_CTR", 0, unit=None),
            "PKT_LEN": ParsedDataItem("PKT_LEN", 2775, unit=None),
        }
    )

    assert expected_de == decom_test_data[1][0].ccsds_header


def test_bytearrays(decom_test_data):
    for hist_test_data in decom_test_data[0]:
        assert isinstance(hist_test_data.HISTOGRAM_DATA, bytearray)

    for de_test_data in decom_test_data[1]:
        assert isinstance(de_test_data.DE_DATA, bytearray)

    # print(decom_test_data[0][0].HISTOGRAM_DATA[:32].hex())

    # first 32 bytes, from original binary string of the first test histogram packet
    expected_value_hist_partial = bytearray.fromhex(
        "1D1E1E1D1D1E1E1E1E1D1D1E1F1D1E1E1F1D1E1E1F1E1E1E1F1F1E1E1E1F1F1E"
    )

    assert decom_test_data[0][0].HISTOGRAM_DATA[:32] == expected_value_hist_partial

    expected_value_de_partial = bytearray.fromhex(
        "033B8512033B8511001E74D6033B851300010100B71B444400372B0109CB07D7"
    )

    assert decom_test_data[1][0].DE_DATA[:32] == expected_value_de_partial


def test_de_byte_length(decom_test_data):
    """Test expected byte length for direct event data"""
    assert len(decom_test_data[1][0].DE_DATA) == 2764
    assert len(decom_test_data[1][1].DE_DATA) == 2208
    assert len(decom_test_data[1][2].DE_DATA) == 1896
