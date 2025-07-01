import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": 898,
                "filename": "FM45_UltraFM45Extra_TV_Tests_"
                "2024-01-22T0930_20240122T093008.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.external_test_data
def test_image_raw_events_decom(decom_test_data, xtce_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""
    filename = (
        "ultra45_raw_sc_imgpriority1evnt_FM45_UltraFM45Extra_TV_Tests_"
        "2024-01-22T0930_20240122T093008.csv"
    )
    priority_1_events_test_path = (
        imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename
    )

    decom_ultra = decom_test_data
    df = pd.read_csv(priority_1_events_test_path, index_col="MET")

    vars_to_compare = {
        "SID": "sid",
        "Spin": "spin",
        "AbortFlag": "abortflag",
        "StartDelay": "startdelay",
        "Count": "count",
        "CoinType": "coin_type",
        "StartType": "start_type",
        "StopType": "stop_type",
        "StartPosTDC": "start_pos_tdc",
        "StopNorthTDC": "stop_north_tdc",
        "StopEastTDC": "stop_east_tdc",
        "StopSouthTDC": "stop_south_tdc",
        "StopWestTDC": "stop_west_tdc",
        "CoinNorthTDC": "coin_north_tdc",
        "CoinSouthTDC": "coin_south_tdc",
        "CoinDiscreteTDC": "coin_discrete_tdc",
        "EnergyOrPH": "energy_ph",
        "PulseWidth": "pulse_width",
        "PhaseAngle": "phase_angle",
        "Bin": "bin",
    }

    for df_var, xr_var in vars_to_compare.items():
        good_values = df[df_var].values != -1
        np.testing.assert_array_equal(
            df[df_var].values[good_values], decom_ultra[xr_var].values[good_values]
        )
