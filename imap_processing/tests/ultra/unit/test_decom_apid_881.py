import json

import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.ultra.l0.ultra_utils import RATES_KEYS, ULTRA_RATES


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_RATES.apid[0],
                "filename": "Ultra45_EM_SwRI_Cal_Run7_"
                "ThetaScan_20220530T225054.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_image_rate_decom(decom_test_data, rates_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""
    decom_ultra, _ = decom_test_data

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


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_RATES.apid[0],
                "filename": "FM45_UltraFM45_Functional_2024-01-22T0105_"
                "20240122T010548.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_image_rate_decom_zero_width(decom_test_data):
    """This function tests for cases in which the width is zero within the packet."""
    test_path = (
        imap_module_directory
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / "ultra45_raw_sc_ultraimgrates_20220530_00.csv"
    )

    decom_ultra, _ = decom_test_data

    df = pd.read_csv(test_path, index_col="MET")
    total_packets = 163

    np.testing.assert_array_equal(df.SID, decom_ultra["SID"])
    np.testing.assert_array_equal(df.Spin, decom_ultra["SPIN"])
    np.testing.assert_array_equal(df.AbortFlag, decom_ultra["ABORTFLAG"])
    np.testing.assert_array_equal(df.StartDelay, decom_ultra["STARTDELAY"])

    for i in range(total_packets):
        t = int(df["SequenceCount"].iloc[i])  # Ensure we get an integer value
        expected_arr = json.loads(df.loc[df["SequenceCount"] == t, "Counts"].values[0])
        arr = []
        for name in RATES_KEYS:
            arr.append(decom_ultra[name][i])
        assert expected_arr == arr
