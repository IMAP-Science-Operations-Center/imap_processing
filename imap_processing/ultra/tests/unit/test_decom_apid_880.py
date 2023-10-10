import pandas as pd
import pytest

from imap_processing import decom


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


def test_aux_decom(decom_test_data, xtce_aux_test_path):
    """This function reads validation data and checks that
    decom data matches validation data for auxiliary packet"""

    df = pd.read_csv(xtce_aux_test_path, index_col="MET")

    for packet in decom_test_data:
        time = packet.data["SHCOARSE"].derived_value
        subdf = df.loc[time]

        assert subdf.SpinStartSeconds == packet.data["TIMESPINSTART"].derived_value
        assert (
            subdf.SpinStartSubseconds == packet.data["TIMESPINSTARTSUB"].derived_value
        )
        assert subdf.SpinDuration == packet.data["DURATION"].derived_value
        assert subdf.SpinNumber == packet.data["SPINNUMBER"].derived_value
        assert subdf.SpinDataTime == packet.data["TIMESPINDATA"].derived_value
        assert subdf.SpinPeriod == packet.data["SPINPERIOD"].derived_value
        assert subdf.SpinPhase == packet.data["SPINPHASE"].derived_value
