import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.ultra_utils import ULTRA_EVENTS


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_EVENTS.apid[0],
                "filename": "FM45_7P_Phi0.0_BeamCal_LinearScan_phi0.04"
                "_theta-0.01_20230821T121304.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_image_raw_events_decom(
    decom_test_data, events_test_path, ccsds_path_events, xtce_path
):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""
    decom_ultra = decom_test_data

    df = pd.read_csv(events_test_path, index_col="MET")
    df.replace(-1, np.iinfo(np.int64).min, inplace=True)

    np.testing.assert_array_equal(df.SID, decom_ultra["sid"])
    np.testing.assert_array_equal(df["Spin"], decom_ultra["spin"])
    np.testing.assert_array_equal(df["AbortFlag"], decom_ultra["abortflag"])
    np.testing.assert_array_equal(df["StartDelay"], decom_ultra["startdelay"])
    np.testing.assert_array_equal(df["Count"], decom_ultra["count"])
    np.testing.assert_array_equal(df["CoinType"], decom_ultra["coin_type"])
    np.testing.assert_array_equal(df["StartType"], decom_ultra["start_type"])
    np.testing.assert_array_equal(df["StopType"], decom_ultra["stop_type"])
    np.testing.assert_array_equal(df["StartPosTDC"], decom_ultra["start_pos_tdc"])
    np.testing.assert_array_equal(df["StopNorthTDC"], decom_ultra["stop_north_tdc"])
    np.testing.assert_array_equal(df["StopEastTDC"], decom_ultra["stop_east_tdc"])
    np.testing.assert_array_equal(df["StopSouthTDC"], decom_ultra["stop_south_tdc"])
    np.testing.assert_array_equal(df["StopWestTDC"], decom_ultra["stop_west_tdc"])
    np.testing.assert_array_equal(df["CoinNorthTDC"], decom_ultra["coin_north_tdc"])
    np.testing.assert_array_equal(df["CoinSouthTDC"], decom_ultra["coin_south_tdc"])
    np.testing.assert_array_equal(
        df["CoinDiscreteTDC"], decom_ultra["coin_discrete_tdc"]
    )
    np.testing.assert_array_equal(df["EnergyOrPH"], decom_ultra["energy_ph"])
    np.testing.assert_array_equal(df["PulseWidth"], decom_ultra["pulse_width"])
    np.testing.assert_array_equal(df["PhaseAngle"], decom_ultra["phase_angle"])
    np.testing.assert_array_equal(df["Bin"], decom_ultra["bin"])


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_EVENTS.apid[0],
                "filename": "FM45_7P_Phi0.0_BeamCal_LinearScan_phi0.04"
                "_theta-0.01_20230821T121304.CCSDS",
            }
        )
    ],
    indirect=True,
)
def test_image_raw_events_decom_flags(decom_test_data, events_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    decom_ultra = decom_test_data
    df = pd.read_csv(events_test_path, index_col="MET")
    df.replace(-1, np.iinfo(np.int64).min, inplace=True)

    np.testing.assert_array_equal(df["CnT"], decom_ultra["event_flag_cnt"])
    np.testing.assert_array_equal(df["PHCmpSL"], decom_ultra["event_flag_phcmpsl"])
    np.testing.assert_array_equal(df["PHCmpSR"], decom_ultra["event_flag_phcmpsr"])
    np.testing.assert_array_equal(df["PHCmpCD"], decom_ultra["event_flag_phcmpcd"])
    np.testing.assert_array_equal(df["SSDS7"], decom_ultra["ssd_flag_7"])
    np.testing.assert_array_equal(df["SSDS6"], decom_ultra["ssd_flag_6"])
    np.testing.assert_array_equal(df["SSDS5"], decom_ultra["ssd_flag_5"])
    np.testing.assert_array_equal(df["SSDS4"], decom_ultra["ssd_flag_4"])
    np.testing.assert_array_equal(df["SSDS3"], decom_ultra["ssd_flag_3"])
    np.testing.assert_array_equal(df["SSDS2"], decom_ultra["ssd_flag_2"])
    np.testing.assert_array_equal(df["SSDS1"], decom_ultra["ssd_flag_1"])
    np.testing.assert_array_equal(df["SSDS0"], decom_ultra["ssd_flag_0"])
    np.testing.assert_array_equal(df["CFDCoinTN"], decom_ultra["cfd_flag_cointn"])
    np.testing.assert_array_equal(df["CFDCoinBN"], decom_ultra["cfd_flag_coinbn"])
    np.testing.assert_array_equal(df["CFDCoinTS"], decom_ultra["cfd_flag_coints"])
    np.testing.assert_array_equal(df["CFDCoinBS"], decom_ultra["cfd_flag_coinbs"])
    np.testing.assert_array_equal(df["CFDCoinD"], decom_ultra["cfd_flag_coind"])
    np.testing.assert_array_equal(df["CFDStartRF"], decom_ultra["cfd_flag_startrf"])
    np.testing.assert_array_equal(df["CFDStartLF"], decom_ultra["cfd_flag_startlf"])
    np.testing.assert_array_equal(df["CFDStartRP"], decom_ultra["cfd_flag_startrp"])
    np.testing.assert_array_equal(df["CFDStartLP"], decom_ultra["cfd_flag_startlp"])
    np.testing.assert_array_equal(df["CFDStopTN"], decom_ultra["cfd_flag_stoptn"])
    np.testing.assert_array_equal(df["CFDStopBN"], decom_ultra["cfd_flag_stopbn"])
    np.testing.assert_array_equal(df["CFDStopTE"], decom_ultra["cfd_flag_stopte"])
    np.testing.assert_array_equal(df["CFDStopBE"], decom_ultra["cfd_flag_stopbe"])
    np.testing.assert_array_equal(df["CFDStopTS"], decom_ultra["cfd_flag_stopts"])
    np.testing.assert_array_equal(df["CFDStopBS"], decom_ultra["cfd_flag_stopbs"])
    np.testing.assert_array_equal(df["CFDStopTW"], decom_ultra["cfd_flag_stoptw"])
    np.testing.assert_array_equal(df["CFDStopBW"], decom_ultra["cfd_flag_stopbw"])
