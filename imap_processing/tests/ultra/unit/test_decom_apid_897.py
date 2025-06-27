import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_ENERGY_EVENTS


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_ENERGY_EVENTS.apid[0],
                "filename": "FM45_7P_Phi0.0_BeamCal_LinearScan_phi0.04"
                "_theta-0.01_20230821T121304.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_image_raw_energy_events_decom(
    decom_test_data, events_test_path, ccsds_path_events, xtce_path
):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""
    decom_ultra = decom_test_data

    df = pd.read_csv(events_test_path, index_col="MET")

    # # Check all values of each column are as expected,
    # except for those set to fill value
    np.testing.assert_array_equal(
        df["StopType"].values[df["StopType"].values != -1],
        decom_ultra["stop_type"].values[df["StopType"].values != -1],
    )
    np.testing.assert_array_equal(
        df["EnergyOrPH"].values[df["EnergyOrPH"].values != -1],
        decom_ultra["energy_ph"].values[df["EnergyOrPH"].values != -1],
    )
    np.testing.assert_array_equal(
        df["PulseWidth"].values[df["PulseWidth"].values != -1],
        decom_ultra["pulse_width"].values[df["PulseWidth"].values != -1],
    )
    np.testing.assert_array_equal(
        df["Bin"].values[df["Bin"].values != -1],
        decom_ultra["bin"].values[df["Bin"].values != -1],
    )
