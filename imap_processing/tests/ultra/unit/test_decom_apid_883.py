import json

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import ULTRA_TOF


@pytest.fixture()
def decom_ultra(ccsds_path_tof, xtce_path):
    """Data for decom_ultra"""
    data_packet_list = decom_ultra_apids(ccsds_path_tof, xtce_path, ULTRA_TOF.apid[0])
    return data_packet_list


def test_tof_decom(decom_ultra, tof_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(tof_test_path, index_col="SequenceCount")

    np.testing.assert_array_equal(df.Spin, decom_ultra["SPIN"])
    np.testing.assert_array_equal(df.AbortFlag, decom_ultra["ABORTFLAG"])
    np.testing.assert_array_equal(df.StartDelay, decom_ultra["STARTDELAY"])
    assert json.loads(df["P00s"].values[0])[0] == decom_ultra["P00"][0]

    for count in df.index.get_level_values("SequenceCount").values:
        df_data = df[df.index.get_level_values("SequenceCount") == count].Images.values[
            0
        ]
        index = decom_ultra["SRC_SEQ_CTR"].index(count)
        decom_data = decom_ultra["PACKETDATA"][index]
        df_data_array = np.array(json.loads(df_data)[0])

        np.testing.assert_array_equal(df_data_array, decom_data)
