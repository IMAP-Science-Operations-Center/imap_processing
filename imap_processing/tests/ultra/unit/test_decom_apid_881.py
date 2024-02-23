import json

import numpy as np
import pandas as pd
import pytest

from imap_processing import decom
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import ULTRA_RATES


@pytest.fixture()
def decom_test_data(ccsds_path, xtce_path):
    """Data for decom"""
    data_packet_list = decom.decom_packets(ccsds_path, xtce_path)
    return data_packet_list


@pytest.fixture()
def decom_ultra(ccsds_path, xtce_path):
    """Data for decom_ultra"""
    data_packets = decom_ultra_apids(ccsds_path, xtce_path, ULTRA_RATES.value.apid[0])
    return data_packets


def test_image_rate_decom(decom_ultra, rates_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(rates_test_path, index_col="MET")
    count = 0
    total_packets = 23

    np.testing.assert_array_equal(df.SID, decom_ultra["SID"])
    np.testing.assert_array_equal(df.Spin, decom_ultra["SPIN"])
    np.testing.assert_array_equal(df.AbortFlag, decom_ultra["ABORTFLAG"])
    np.testing.assert_array_equal(df.StartDelay, decom_ultra["STARTDELAY"])

    for time in decom_ultra["SHCOARSE"]:
        arr1 = json.loads(df.loc[time].Counts)
        index = decom_ultra["SHCOARSE"].index(time)
        arr2 = decom_ultra["FASTDATA_00"][index]
        np.testing.assert_array_equal(arr1, arr2)
        count += 1

    assert count == total_packets
