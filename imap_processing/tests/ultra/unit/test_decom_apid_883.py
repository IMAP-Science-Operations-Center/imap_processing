import json

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_TOF

# TODO: discuss with instrument team incomplete set of SIDs


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_TOF.apid[0],
                "filename": "FM45_TV_Cycle6_Hot_Ops_" "Front212_20240124T063837.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_tof_decom(decom_test_data, tof_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    decom_ultra, _ = decom_test_data
    df = pd.read_csv(tof_test_path, index_col="SequenceCount")

    np.testing.assert_array_equal(df.Spin, decom_ultra["SPIN"].flatten())
    np.testing.assert_array_equal(df.AbortFlag, decom_ultra["ABORTFLAG"].flatten())
    np.testing.assert_array_equal(df.StartDelay, decom_ultra["STARTDELAY"].flatten())
    assert json.loads(df["P00s"].values[0])[0] == decom_ultra["P00"][0][0]

    for count in df.index.get_level_values("SequenceCount").values:
        df_data = df[df.index.get_level_values("SequenceCount") == count].Images.values[
            0
        ]
        rows, cols = np.where(decom_ultra["SRC_SEQ_CTR"] == count)
        decom_data = decom_ultra["PACKETDATA"][rows[0]][cols[0]]
        df_data_array = np.array(json.loads(df_data)[0])

        np.testing.assert_array_equal(df_data_array, decom_data)
