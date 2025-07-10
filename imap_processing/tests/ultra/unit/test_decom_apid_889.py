import json

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.decom_ultra import process_ultra_energy_spectra
from imap_processing.ultra.l0.ultra_utils import (
    ENERGY_SPECTRA_KEYS,
    ULTRA_ENERGY_SPECTRA,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.mark.external_test_data
def test_energy_spectra_decom(xtce_path, energy_spectra_test_path, ccsds_path_startup):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    datasets_by_apid = packet_file_to_datasets(ccsds_path_startup, xtce_path)
    decom_ultra = process_ultra_energy_spectra(
        datasets_by_apid[ULTRA_ENERGY_SPECTRA.apid[1]]
    )

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
