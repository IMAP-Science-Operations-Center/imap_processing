import json

import numpy as np
import pandas as pd
import pytest

from imap_processing import decom
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import RATES_KEYS, ULTRA_RATES


@pytest.fixture()
def decom_test_data(ccsds_path, xtce_path):
    """Data for decom"""
    data_packet_list = decom.decom_packets(ccsds_path, xtce_path)
    return data_packet_list


@pytest.fixture()
def decom_ultra(ccsds_path, xtce_path):
    """Data for decom_ultra"""
    data_packets = decom_ultra_apids(ccsds_path, xtce_path, ULTRA_RATES.apid[0])
    return data_packets


def test_image_rate_decom(decom_ultra, rates_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(rates_test_path, index_col="MET")
    total_packets = 23

    np.testing.assert_array_equal(df.SID, decom_ultra["SID"])
    np.testing.assert_array_equal(df.Spin, decom_ultra["SPIN"])
    np.testing.assert_array_equal(df.AbortFlag, decom_ultra["ABORTFLAG"])
    np.testing.assert_array_equal(df.StartDelay, decom_ultra["STARTDELAY"])

    # Spot-check first packet
    t0 = decom_ultra["SHCOARSE"][0]
    expected_arr0 = json.loads(df.loc[t0].Counts)
    arr = []
    for name in RATES_KEYS:
        arr.append(decom_ultra[name][0])
    assert expected_arr0 == arr

    # Spot-check last packet
    tn = decom_ultra["SHCOARSE"][total_packets - 1]
    expected_arrn = json.loads(df.loc[tn].Counts)
    arr = []
    for name in RATES_KEYS:
        arr.append(decom_ultra[name][total_packets - 1])
    assert expected_arrn == arr
