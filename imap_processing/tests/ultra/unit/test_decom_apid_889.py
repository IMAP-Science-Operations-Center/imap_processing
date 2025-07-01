import json

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import (
    ENERGY_SPECTRA_KEYS,
    ULTRA_ENERGY_SPECTRA,
)


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_ENERGY_SPECTRA.apid[1],
                "filename": "FM90_Startup_20230711T081655.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.external_test_data
def test_energy_spectra_decom(decom_test_data, energy_spectra_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""
    decom_ultra = decom_test_data

    df = pd.read_csv(energy_spectra_test_path, index_col="MET")
    total_packets = 26

    np.testing.assert_array_equal(df.SID, decom_ultra["sid"])
    np.testing.assert_array_equal(df.Spin, decom_ultra["spin"])
    np.testing.assert_array_equal(df.AbortFlag, decom_ultra["abortflag"])
    np.testing.assert_array_equal(df.StartDelay, decom_ultra["startdelay"])

    # Spot-check first packet
    t0 = decom_ultra["shcoarse"][0]
    expected_arr0 = json.loads(df.loc[int(t0)].SSDSum)[0]
    assert np.array_equal(expected_arr0, decom_ultra[ENERGY_SPECTRA_KEYS[0]].values[0])

    # Spot-check last packet
    tn = decom_ultra["shcoarse"][total_packets - 1]
    expected_arrn = json.loads(df.loc[int(tn)].SSDSum)[0]
    assert np.array_equal(expected_arrn, decom_ultra[ENERGY_SPECTRA_KEYS[0]].values[-1])
