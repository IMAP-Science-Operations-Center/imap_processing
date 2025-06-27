import json

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ENERGY_RATES_KEYS, ULTRA_ENERGY_RATES


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_ENERGY_RATES.apid[0],
                "filename": "FM45_UltraFM45_Functional_"
                "2024-01-22T0105_20240122T010548.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.external_test_data
def test_image_rate_decom(decom_test_data, energy_rates_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""
    decom_ultra = decom_test_data

    df = pd.read_csv(energy_rates_test_path, index_col="MET")
    total_packets = 315

    np.testing.assert_array_equal(df.SID, decom_ultra["sid"])
    np.testing.assert_array_equal(df.Spin, decom_ultra["spin"])
    np.testing.assert_array_equal(df.AbortFlag, decom_ultra["abortflag"])
    np.testing.assert_array_equal(df.StartDelay, decom_ultra["startdelay"])

    # Spot-check first packet
    t0 = decom_ultra["shcoarse"][0]
    expected_arr0 = json.loads(df.loc[int(t0)].Counts)
    arr = []
    for name in ENERGY_RATES_KEYS:
        arr.append(decom_ultra[name][0])
    assert expected_arr0 == arr

    # Spot-check last packet
    tn = decom_ultra["shcoarse"][total_packets - 1]
    expected_arrn = json.loads(df.loc[int(tn)].Counts)
    arr = []
    for name in ENERGY_RATES_KEYS:
        arr.append(decom_ultra[name][total_packets - 1])
    assert expected_arrn == arr
