import json

import numpy as np
import pandas as pd
import pytest

from imap_processing import decom
from imap_processing.ultra.l0.decom_ultra import decom_ultra_packets


@pytest.fixture()
def decom_test_data(ccsds_path, xtce_image_rates_path):
    """Data for decom"""
    data_packet_list = decom.decom_packets(ccsds_path, xtce_image_rates_path)
    return data_packet_list


@pytest.fixture()
def decom_ultra(ccsds_path, xtce_image_rates_path):
    """Data for decom_ultra"""
    data_packet_list = decom_ultra_packets(ccsds_path, xtce_image_rates_path)
    return data_packet_list


def test_image_rate_length(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 23
    assert len(decom_test_data) == total_packets


def test_image_rate_decom(decom_ultra, xtce_image_rates_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(xtce_image_rates_test_path, index_col="MET")

    assert (df.SID == decom_ultra["science_id"]).all()
    assert (df.Spin == decom_ultra["spin_data"]).all()
    assert (df.AbortFlag == decom_ultra["abortflag_data"]).all()
    assert (df.StartDelay == decom_ultra["startdelay_data"]).all()

    for time in decom_ultra.epoch.values:
        arr1 = json.loads(df.loc[time].Counts)
        arr2 = decom_ultra["fastdata_00"].sel(epoch=time).data
        np.testing.assert_array_equal(arr1, arr2)
