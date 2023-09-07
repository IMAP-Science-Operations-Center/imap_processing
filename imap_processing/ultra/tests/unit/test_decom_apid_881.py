import ast

import numpy as np
import pandas as pd
import pytest

from imap_processing import decom
from imap_processing.ultra.decom_ultra import decom_ultra_packets


@pytest.fixture()
def decom_test_data(ccsds_path, xtce_image_rates_path):
    """Data for decom"""
    data_packet_list = decom.decom_packets(ccsds_path, xtce_image_rates_path)
    return data_packet_list

@pytest.fixture()
def decom_ultra(ccsds_path, xtce_image_rates_path):
    """Data for decom_ultra"""
    data_packet_list = decom_ultra_packets(ccsds_path,  xtce_image_rates_path)
    return data_packet_list

def test_ultra_apid_881_length(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 22
    assert len(decom_test_data) == total_packets

def test_ultra_apid_881(decom_ultra, xtce_image_rates_test_path):
    """Test values for apid 881"""

    df = pd.read_csv(xtce_image_rates_test_path, index_col='MET')

    for time in decom_ultra.time.values:
        assert df.loc[time].SID == decom_ultra['sid_data'].sel(time=time)
        assert df.loc[time].Spin == decom_ultra['spin_data'].sel(time=time)
        assert df.loc[time].AbortFlag == decom_ultra['abortflag_data'].sel(time=time)
        assert df.loc[time].StartDelay == decom_ultra['startdelay_data'].sel(time=time)

        arr1 = ast.literal_eval(df.loc[time].Counts)
        arr2 = decom_ultra['fastdata_00'].sel(time=time).data

        np.testing.assert_array_equal(arr1, arr2.item()[0:48])
