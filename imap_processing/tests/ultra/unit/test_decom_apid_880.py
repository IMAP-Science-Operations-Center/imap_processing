import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_AUX
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture
def decom_ultra_aux_data(ccsds_path, xtce_path):
    """Data for decom_ultra_aux"""

    datasets_by_apid = packet_file_to_datasets(ccsds_path, xtce_path)
    decom_ultra_dataset = datasets_by_apid[ULTRA_AUX.apid[0]]
    return decom_ultra_dataset


def test_aux_modes(decom_ultra_aux_data):
    """Test if enumerated values derived correctly"""

    assert np.all(decom_ultra_aux_data["spinperiodvalid"] == 0)
    assert np.all(decom_ultra_aux_data["spinphasevalid"] == 1)
    assert np.all(decom_ultra_aux_data["spinperiodsource"] == 1)
    assert np.all(decom_ultra_aux_data["catbedheaterflag"] == 0)
    assert np.all(decom_ultra_aux_data["hwmode"] == 0)
    assert np.all(decom_ultra_aux_data["imcenb"] == 0)
    assert np.all(decom_ultra_aux_data["leftdeflectioncharge"] == 0)
    assert np.all(decom_ultra_aux_data["rightdeflectioncharge"] == 0)

    assert len(decom_ultra_aux_data["shcoarse"]) == 23


def test_aux_decom(decom_ultra_aux_data, aux_test_path):
    """This function reads validation data and checks that
    decom data matches validation data for auxiliary packet"""

    df = pd.read_csv(aux_test_path, index_col="MET")

    np.testing.assert_array_equal(
        df.SpinStartSeconds, decom_ultra_aux_data["timespinstart"]
    )
    np.testing.assert_array_equal(
        df.SpinStartSubseconds, decom_ultra_aux_data["timespinstartsub"]
    )
    np.testing.assert_array_equal(df.SpinDuration, decom_ultra_aux_data["duration"])
    np.testing.assert_array_equal(df.SpinNumber, decom_ultra_aux_data["spinnumber"])
    np.testing.assert_array_equal(df.SpinDataTime, decom_ultra_aux_data["timespindata"])
    np.testing.assert_array_equal(df.SpinPeriod, decom_ultra_aux_data["spinperiod"])
    np.testing.assert_array_equal(df.SpinPhase, decom_ultra_aux_data["spinphase"])
