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
    packets = decom.decom_packets(ccsds_path, xtce_path)
    grouped_data = group_by_apid(packets)
    data = {ULTRA_AUX.apid[0]: grouped_data[ULTRA_AUX.apid[0]]}

    data_packet_list = decom_ultra_apids(data, ULTRA_AUX.apid[0])
    return data_packet_list, packets


def test_aux_enumerated(decom_test_data):
    """Test if enumerated values derived correctly"""

    _, packets = decom_test_data

    count = 0  # count number of packets with APID 880
    total_packets = 23

    grouped_data = group_by_apid(packets)
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

    data_packet_list, packets = decom_test_data

    for packet in packets:
        if packet.header["PKT_APID"].derived_value == 880:
            assert packet.data["HWMODE"].derived_value == "MODE0"
            assert packet.data["IMCENB"].derived_value == "MODE0"
            assert packet.data["LEFTDEFLECTIONCHARGE"].derived_value == "MODE0"
            assert packet.data["RIGHTDEFLECTIONCHARGE"].derived_value == "MODE0"


def test_aux_decom(decom_test_data, aux_test_path):
    """This function reads validation data and checks that
    decom data matches validation data for auxiliary packet"""

    data_packet_list, packets = decom_test_data

    df = pd.read_csv(aux_test_path, index_col="MET")

    np.testing.assert_array_equal(
        df.SpinStartSeconds, data_packet_list["TIMESPINSTART"]
    )
    np.testing.assert_array_equal(
        df.SpinStartSubseconds, data_packet_list["TIMESPINSTARTSUB"]
    )
    np.testing.assert_array_equal(df.SpinDuration, data_packet_list["DURATION"])
    np.testing.assert_array_equal(df.SpinNumber, data_packet_list["SPINNUMBER"])
    np.testing.assert_array_equal(df.SpinDataTime, data_packet_list["TIMESPINDATA"])
    np.testing.assert_array_equal(df.SpinPeriod, data_packet_list["SPINPERIOD"])
    np.testing.assert_array_equal(df.SpinPhase, data_packet_list["SPINPHASE"])
