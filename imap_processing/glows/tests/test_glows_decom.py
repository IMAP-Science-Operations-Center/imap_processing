import logging
from pathlib import Path

import pytest

from imap_processing.glows.l0 import glows_decom

logging.basicConfig(level=logging.ERROR)


@pytest.fixture()
def decom_test_data():
    """Read test data from file"""
    current_directory = Path(__file__).parent
    packet_path = f"{current_directory}/glows_test_packet_20230920_v00.pkts"
    data_packet_list = glows_decom.decom_packets(packet_path)
    return data_packet_list


def test_glows_decom_count(decom_test_data):
    expected_hist_len = 61
    expected_de_len = 311

    assert len(decom_test_data) == 2

    # Histogram data
    assert len(decom_test_data[0]) == expected_hist_len

    # Direct events data
    assert len(decom_test_data[1]) == expected_de_len


def test_glows_hist_data(decom_test_data):
    for hist in decom_test_data[0]:
        for key in hist.data:
            assert hist.data[key] is not None
        assert len(hist.data["HISTOGRAM_DATA"]) == 3599
        assert min(hist.data["HISTOGRAM_DATA"]) > 10
        assert max(hist.data["HISTOGRAM_DATA"]) < 170

    expected_data = {
        "SHCOARSE": 54232338,
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
        assert decom_test_data[0][0].data[key] == expected_data[key]


def test_glows_de_data(decom_test_data):
    for de in decom_test_data[1]:
        for key in de.data:
            assert de.data[key] is not None

    expected_data = {"SHCOARSE": 54233694, "SEC": 54232338, "LEN": 1, "SEQ": 0}
    for key in expected_data.keys():
        assert decom_test_data[1][0].data[key] == expected_data[key]
