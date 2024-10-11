import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_AUX
from imap_processing.utils import group_by_apid


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_AUX.apid[0],
                "filename": "Ultra45_EM_SwRI_Cal_Run7_"
                "ThetaScan_20220530T225054.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_aux_enumerated(decom_test_data):
    """Test if enumerated values derived correctly"""

    _, packets = decom_test_data

    count = 0  # count number of packets with APID 880
    total_packets = 23

    grouped_data = group_by_apid(packets)
    apid_data = grouped_data[880]

    for packet in apid_data:
        assert packet["SPINPERIODVALID"] == "INVALID"
        assert packet["SPINPHASEVALID"] == "VALID"
        assert packet["SPINPERIODSOURCE"] == "NOMINAL"
        assert packet["CATBEDHEATERFLAG"] == "UNFLAGGED"
        count += 1

    assert count == total_packets


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_AUX.apid[0],
                "filename": "Ultra45_EM_SwRI_Cal_Run7_"
                "ThetaScan_20220530T225054.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_aux_mode(decom_test_data):
    """Test if enumerated values derived correctly"""

    _, packets = decom_test_data

    for packet in packets:
        if packet["PKT_APID"] == 880:
            assert packet["HWMODE"] == "MODE0"
            assert packet["IMCENB"] == "MODE0"
            assert packet["LEFTDEFLECTIONCHARGE"] == "MODE0"
            assert packet["RIGHTDEFLECTIONCHARGE"] == "MODE0"


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_AUX.apid[0],
                "filename": "Ultra45_EM_SwRI_Cal_Run7_"
                "ThetaScan_20220530T225054.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_aux_decom(decom_test_data, aux_test_path):
    """This function reads validation data and checks that
    decom data matches validation data for auxiliary packet"""

    decom_ultra, _ = decom_test_data

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
