import json

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_PHXTOF_HIGH_ENERGY


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_PHXTOF_HIGH_ENERGY.apid[0],
                "filename": "FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_"
                "20240122T093008.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.external_test_data
def test_phxtof_high_energy_decom(decom_test_data, tof_high_energy_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    decom_ultra = decom_test_data
    df = pd.read_csv(tof_high_energy_test_path, index_col="SequenceCount")

    np.testing.assert_array_equal(df.Spin, decom_ultra["spin"].values.flatten())
    np.testing.assert_array_equal(
        df.AbortFlag, decom_ultra["abortflag"].values.flatten()
    )
    np.testing.assert_array_equal(
        df.StartDelay, decom_ultra["startdelay"].values.flatten()
    )

    for count in df.index.get_level_values("SequenceCount").values:
        df_data = df[
            df.index.get_level_values("SequenceCount") == count
        ].UltraImage.values[0]
        rows, cols = np.where(decom_ultra["src_seq_ctr"] == count)
        decom_data = decom_ultra["packetdata"][rows[0]][cols[0]]
        df_data_array = np.array(json.loads(df_data)[0])

        np.testing.assert_array_equal(df_data_array, decom_data)
