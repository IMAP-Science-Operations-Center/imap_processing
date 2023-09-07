import pandas as pd
import pytest

from imap_processing import decom


@pytest.fixture()
def decom_test_data(ccsds_path, xtce_aux_path):
    """Read test data from file"""
    data_packet_list = decom.decom_packets(ccsds_path, xtce_aux_path)
    return data_packet_list


def test_ultra_apid_880_length(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 23
    assert len(decom_test_data) == total_packets


def test_enumerated_apid_880(decom_test_data):
    """Test if enumerated values derived correctly"""

    for packet in decom_test_data:

        if packet.data["SPINPERIODVALID"].raw_value == 0:
            assert packet.data["SPINPERIODVALID"].derived_value == 'INVALID'
        elif packet.data["SPINPERIODVALID"].raw_value == 1:
            assert packet.data["SPINPERIODVALID"].derived_value == 'VALID'
        if packet.data["SPINPHASEVALID"].raw_value == 0:
            assert packet.data["SPINPHASEVALID"].derived_value == 'INVALID'
        elif packet.data["SPINPHASEVALID"].raw_value == 1:
            assert packet.data["SPINPHASEVALID"].derived_value == 'VALID'
        if packet.data["SPINPERIODSOURCE"].raw_value == 0:
            assert packet.data["SPINPERIODSOURCE"].derived_value == 'SAFING'
        elif packet.data["SPINPERIODSOURCE"].raw_value == 1:
            assert packet.data["SPINPERIODSOURCE"].derived_value == 'NOMINAL'
        if packet.data["CATBEDHEATERFLAG"].raw_value == 0:
            assert packet.data["CATBEDHEATERFLAG"].derived_value == 'UNFLAGGED'
        if packet.data["HWMODE"].raw_value == 0:
            assert packet.data["HWMODE"].derived_value == 'MODE0'
        elif packet.data["HWMODE"].raw_value == 1:
            assert packet.data["HWMODE"].derived_value == 'MODE1'
        if packet.data["IMCENB"].raw_value == 0:
            assert packet.data["IMCENB"].derived_value == 'MODE0'
        elif packet.data["IMCENB"].raw_value == 1:
            assert packet.data["IMCENB"].derived_value == 'MODE1'
        if packet.data["LEFTDEFLECTIONCHARGE"].raw_value == 0:
            assert packet.data["LEFTDEFLECTIONCHARGE"].derived_value == 'MODE0'
        elif packet.data["LEFTDEFLECTIONCHARGE"].raw_value == 1:
            assert packet.data["LEFTDEFLECTIONCHARGE"].derived_value == 'MODE1'
        if packet.data["RIGHTDEFLECTIONCHARGE"].raw_value == 0:
            assert packet.data["RIGHTDEFLECTIONCHARGE"].derived_value == 'MODE0'
        elif packet.data["RIGHTDEFLECTIONCHARGE"].raw_value == 1:
            assert packet.data["RIGHTDEFLECTIONCHARGE"].derived_value == 'MODE1'

def test_ultra_apid_880(decom_test_data, xtce_aux_test_path):
    """Test values for apid 880"""

    df = pd.read_csv(xtce_aux_test_path, index_col='MET')

    for packet in decom_test_data:

        time = packet.data['SHCOARSE'].derived_value

        assert df.loc[time].SpinStartSeconds == \
               packet.data["TIMESPINSTART"].derived_value
        assert df.loc[time].SpinStartSubseconds == \
               packet.data["TIMESPINSTARTSUB"].derived_value
        assert df.loc[time].SpinDuration == packet.data["DURATION"].derived_value
        assert df.loc[time].SpinNumber == packet.data["SPINNUMBER"].derived_value
        assert df.loc[time].SpinDataTime == packet.data["TIMESPINDATA"].derived_value
        assert df.loc[time].SpinPeriod == packet.data["SPINPERIOD"].derived_value
        assert df.loc[time].SpinPhase == packet.data["SPINPHASE"].derived_value
