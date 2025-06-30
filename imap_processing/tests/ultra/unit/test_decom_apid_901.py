import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": 901,
                "filename": "FM45_UltraFM45Extra_TV_Tests_"
                "2024-01-22T0930_20240122T093008.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.external_test_data
def test_image_raw_events_decom(decom_test_data, ccsds_path_events, xtce_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""
    filename = (
        "ultra45_raw_sc_imgpriority4evnt_FM45_UltraFM45Extra_TV_Tests_"
        "2024-01-22T0930_20240122T093008.csv"
    )
    priority_4_events_test_path = (
        imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename
    )

    decom_ultra = decom_test_data

    df = pd.read_csv(priority_4_events_test_path, index_col="MET")

    # # Check all values of each column are as expected,
    # except for those set to fill value
    np.testing.assert_array_equal(
        df["SID"].values[df["SID"].values != -1],
        decom_ultra["sid"].values[df["SID"].values != -1],
    )
    np.testing.assert_array_equal(
        df["Spin"].values[df["Spin"].values != -1],
        decom_ultra["spin"].values[df["Spin"].values != -1],
    )
    np.testing.assert_array_equal(
        df["AbortFlag"].values[df["AbortFlag"].values != -1],
        decom_ultra["abortflag"].values[df["AbortFlag"].values != -1],
    )
    np.testing.assert_array_equal(
        df["StartDelay"].values[df["StartDelay"].values != -1],
        decom_ultra["startdelay"].values[df["StartDelay"].values != -1],
    )
    np.testing.assert_array_equal(
        df["Count"].values[df["Count"].values != -1],
        decom_ultra["count"].values[df["Count"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CoinType"].values[df["CoinType"].values != -1],
        decom_ultra["coin_type"].values[df["CoinType"].values != -1],
    )
    np.testing.assert_array_equal(
        df["StartType"].values[df["StartType"].values != -1],
        decom_ultra["start_type"].values[df["StartType"].values != -1],
    )
    np.testing.assert_array_equal(
        df["StopType"].values[df["StopType"].values != -1],
        decom_ultra["stop_type"].values[df["StopType"].values != -1],
    )
    np.testing.assert_array_equal(
        df["StartPosTDC"].values[df["StartPosTDC"].values != -1],
        decom_ultra["start_pos_tdc"].values[df["StartPosTDC"].values != -1],
    )
    np.testing.assert_array_equal(
        df["StopNorthTDC"].values[df["StopNorthTDC"].values != -1],
        decom_ultra["stop_north_tdc"].values[df["StopNorthTDC"].values != -1],
    )
    np.testing.assert_array_equal(
        df["StopEastTDC"].values[df["StopEastTDC"].values != -1],
        decom_ultra["stop_east_tdc"].values[df["StopEastTDC"].values != -1],
    )
    np.testing.assert_array_equal(
        df["StopSouthTDC"].values[df["StopSouthTDC"].values != -1],
        decom_ultra["stop_south_tdc"].values[df["StopSouthTDC"].values != -1],
    )
    np.testing.assert_array_equal(
        df["StopWestTDC"].values[df["StopWestTDC"].values != -1],
        decom_ultra["stop_west_tdc"].values[df["StopWestTDC"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CoinNorthTDC"].values[df["CoinNorthTDC"].values != -1],
        decom_ultra["coin_north_tdc"].values[df["CoinNorthTDC"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CoinSouthTDC"].values[df["CoinSouthTDC"].values != -1],
        decom_ultra["coin_south_tdc"].values[df["CoinSouthTDC"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CoinDiscreteTDC"].values[df["CoinDiscreteTDC"].values != -1],
        decom_ultra["coin_discrete_tdc"].values[df["CoinDiscreteTDC"].values != -1],
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
        df["PhaseAngle"].values[df["PhaseAngle"].values != -1],
        decom_ultra["phase_angle"].values[df["PhaseAngle"].values != -1],
    )
    np.testing.assert_array_equal(
        df["Bin"].values[df["Bin"].values != -1],
        decom_ultra["bin"].values[df["Bin"].values != -1],
    )
