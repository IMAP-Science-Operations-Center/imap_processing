import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.ultra.l0.ultra_utils import ULTRA_ENERGY_EVENTS


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_ENERGY_EVENTS.apid[0],
                "filename": "FM45_UltraFM45_Functional_"
                "2024-01-22T0105_20240122T010548.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.external_test_data
def test_image_raw_energy_events_decom(decom_test_data, ccsds_path_events, xtce_path):
    """This function reads validation data and checks that decom data
    matches validation data for the packet"""

    filename = "ultra45_raw_sc_rawnrgevnt_19840122_00.csv"
    energy_events_test_path = (
        imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename
    )

    decom_ultra = decom_test_data

    df = pd.read_csv(energy_events_test_path, index_col="MET")

    # # Check all values of each column are as expected,
    # except for those set to fill value
    np.testing.assert_array_equal(
        df["StopType"].values[df["StopType"].values != -1],
        decom_ultra["stop_type"].values[df["StopType"].values != -1],
    )
    np.testing.assert_array_equal(
        df["EnergyPH"].values[df["EnergyPH"].values != -1],
        decom_ultra["energy_ph"].values[df["EnergyPH"].values != -1],
    )
    np.testing.assert_array_equal(
        df["PulseWidth"].values[df["PulseWidth"].values != -1],
        decom_ultra["pulse_width"].values[df["PulseWidth"].values != -1],
    )
    np.testing.assert_array_equal(
        df["Bin"].values[df["Bin"].values != -1],
        decom_ultra["bin"].values[df["Bin"].values != -1],
    )
