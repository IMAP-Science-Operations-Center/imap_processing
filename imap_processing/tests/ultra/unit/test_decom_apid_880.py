import numpy as np
import pandas as pd
import pytest

from imap_processing import decom
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import ULTRA_AUX
from imap_processing.utils import group_by_apid


@pytest.fixture()
def decom_test_data(ccsds_path, xtce_path):
    """Read test data from file"""
    data_packet_list = decom.decom_packets(ccsds_path, xtce_path)
    return data_packet_list


def test_aux_enumerated(decom_test_data):
    """Test if enumerated values derived correctly"""

    count = 0  # count number of packets with APID 880
    total_packets = 23

    grouped_data = group_by_apid(decom_test_data)
    apid_data = grouped_data[880]

    for packet in apid_data:
        assert packet.data["SPINPERIODVALID"].derived_value == "INVALID"
        assert packet.data["SPINPHASEVALID"].derived_value == "VALID"
        assert packet.data["SPINPERIODSOURCE"].derived_value == "NOMINAL"
        assert packet.data["CATBEDHEATERFLAG"].derived_value == "UNFLAGGED"
        count += 1

    assert count == total_packets


def test_aux_mode(decom_test_data):
    """Test if enumerated values derived correctly"""

    for packet in decom_test_data:
        if packet.header["PKT_APID"].derived_value == 880:
            assert packet.data["HWMODE"].derived_value == "MODE0"
            assert packet.data["IMCENB"].derived_value == "MODE0"
            assert packet.data["LEFTDEFLECTIONCHARGE"].derived_value == "MODE0"
            assert packet.data["RIGHTDEFLECTIONCHARGE"].derived_value == "MODE0"


def test_aux_decom(ccsds_path, xtce_path, aux_test_path):
    """This function reads validation data and checks that
    decom data matches validation data for auxiliary packet"""

    decom_ultra = decom_ultra_apids(ccsds_path, xtce_path, ULTRA_AUX.apid[0])
    df = pd.read_csv(aux_test_path, index_col="MET")

    np.testing.assert_array_equal(df.SpinStartSeconds, decom_ultra["TIMESPINSTART"])
    np.testing.assert_array_equal(
        df.SpinStartSubseconds, decom_ultra["TIMESPINSTARTSUB"]
    )
    np.testing.assert_array_equal(df.SpinDuration, decom_ultra["DURATION"])
    np.testing.assert_array_equal(df.SpinNumber, decom_ultra["SPINNUMBER"])
    np.testing.assert_array_equal(df.SpinDataTime, decom_ultra["TIMESPINDATA"])
    np.testing.assert_array_equal(df.SpinPeriod, decom_ultra["SPINPERIOD"])
    np.testing.assert_array_equal(df.SpinPhase, decom_ultra["SPINPHASE"])
