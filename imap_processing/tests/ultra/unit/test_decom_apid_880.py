import pandas as pd
import pytest

from imap_processing import decom
from imap_processing.ultra.l0.decom_ultra import decom_image_raw_events_packets


@pytest.fixture()
def decom_test_data(ccsds_path, xtce_aux_path):
    """Read test data from file"""
    data_packet_list = decom.decom_packets(ccsds_path, xtce_aux_path)
    return data_packet_list


def test_aux_length(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 23
    assert len(decom_test_data) == total_packets


def test_aux_enumerated(decom_test_data):
    """Test if enumerated values derived correctly"""

    for packet in decom_test_data:
        assert packet.data["SPINPERIODVALID"].derived_value == "INVALID"
        assert packet.data["SPINPHASEVALID"].derived_value == "VALID"
        assert packet.data["SPINPERIODSOURCE"].derived_value == "NOMINAL"
        assert packet.data["CATBEDHEATERFLAG"].derived_value == "UNFLAGGED"


def test_aux_mode(decom_test_data):
    """Test if enumerated values derived correctly"""

    for packet in decom_test_data:
        assert packet.data["HWMODE"].derived_value == "MODE0"
        assert packet.data["IMCENB"].derived_value == "MODE0"
        assert packet.data["LEFTDEFLECTIONCHARGE"].derived_value == "MODE0"
        assert packet.data["RIGHTDEFLECTIONCHARGE"].derived_value == "MODE0"


def test_aux_decom(ccsds_path, xtce_aux_path, aux_test_path):
    """This function reads validation data and checks that
    decom data matches validation data for auxiliary packet"""

    decom_ultra = decom_image_raw_events_packets(ccsds_path, xtce_aux_path)
    df = pd.read_csv(aux_test_path, index_col="MET")

    assert (df.SpinStartSeconds == decom_ultra["TIMESPINSTART"]).all()
    assert (df.SpinStartSubseconds == decom_ultra["TIMESPINSTARTSUB"]).all()
    assert (df.SpinDuration == decom_ultra["DURATION"]).all()
    assert (df.SpinNumber == decom_ultra["SPINNUMBER"]).all()
    assert (df.SpinDataTime == decom_ultra["TIMESPINDATA"]).all()
    assert (df.SpinPeriod == decom_ultra["SPINPERIOD"]).all()
    assert (df.SpinPhase == decom_ultra["SPINPHASE"]).all()
