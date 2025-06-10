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

    # # Check all values of each column are as expected,
    # except for those set to fill value
    decom_ultra = decom_test_data
    df = pd.read_csv(events_test_path, index_col="MET")

    np.testing.assert_array_equal(
        df["CnT"].values[df["CnT"].values != -1],
        decom_ultra["event_flag_cnt"].values[df["CnT"].values != -1],
    )
    np.testing.assert_array_equal(
        df["PHCmpSL"].values[df["PHCmpSL"].values != -1],
        decom_ultra["event_flag_phcmpsl"].values[df["PHCmpSL"].values != -1],
    )
    np.testing.assert_array_equal(
        df["PHCmpSR"].values[df["PHCmpSR"].values != -1],
        decom_ultra["event_flag_phcmpsr"].values[df["PHCmpSR"].values != -1],
    )
    np.testing.assert_array_equal(
        df["PHCmpCD"].values[df["PHCmpCD"].values != -1],
        decom_ultra["event_flag_phcmpcd"].values[df["PHCmpCD"].values != -1],
    )
    np.testing.assert_array_equal(
        df["SSDS7"].values[df["SSDS7"].values != -1],
        decom_ultra["ssd_flag_7"].values[df["SSDS7"].values != -1],
    )
    np.testing.assert_array_equal(
        df["SSDS6"].values[df["SSDS6"].values != -1],
        decom_ultra["ssd_flag_6"].values[df["SSDS6"].values != -1],
    )
    np.testing.assert_array_equal(
        df["SSDS5"].values[df["SSDS5"].values != -1],
        decom_ultra["ssd_flag_5"].values[df["SSDS5"].values != -1],
    )
    np.testing.assert_array_equal(
        df["SSDS4"].values[df["SSDS4"].values != -1],
        decom_ultra["ssd_flag_4"].values[df["SSDS4"].values != -1],
    )
    np.testing.assert_array_equal(
        df["SSDS3"].values[df["SSDS3"].values != -1],
        decom_ultra["ssd_flag_3"].values[df["SSDS3"].values != -1],
    )
    np.testing.assert_array_equal(
        df["SSDS2"].values[df["SSDS2"].values != -1],
        decom_ultra["ssd_flag_2"].values[df["SSDS2"].values != -1],
    )
    np.testing.assert_array_equal(
        df["SSDS1"].values[df["SSDS1"].values != -1],
        decom_ultra["ssd_flag_1"].values[df["SSDS1"].values != -1],
    )
    np.testing.assert_array_equal(
        df["SSDS0"].values[df["SSDS0"].values != -1],
        decom_ultra["ssd_flag_0"].values[df["SSDS0"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDCoinTN"].values[df["CFDCoinTN"].values != -1],
        decom_ultra["cfd_flag_cointn"].values[df["CFDCoinTN"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDCoinBN"].values[df["CFDCoinBN"].values != -1],
        decom_ultra["cfd_flag_coinbn"].values[df["CFDCoinBN"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDCoinTS"].values[df["CFDCoinTS"].values != -1],
        decom_ultra["cfd_flag_coints"].values[df["CFDCoinTS"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDCoinBS"].values[df["CFDCoinBS"].values != -1],
        decom_ultra["cfd_flag_coinbs"].values[df["CFDCoinBS"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDCoinD"].values[df["CFDCoinD"].values != -1],
        decom_ultra["cfd_flag_coind"].values[df["CFDCoinD"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStartRF"].values[df["CFDStartRF"].values != -1],
        decom_ultra["cfd_flag_startrf"].values[df["CFDStartRF"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStartLF"].values[df["CFDStartLF"].values != -1],
        decom_ultra["cfd_flag_startlf"].values[df["CFDStartLF"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStartRP"].values[df["CFDStartRP"].values != -1],
        decom_ultra["cfd_flag_startrp"].values[df["CFDStartRP"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStartLP"].values[df["CFDStartLP"].values != -1],
        decom_ultra["cfd_flag_startlp"].values[df["CFDStartLP"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStopTN"].values[df["CFDStopTN"].values != -1],
        decom_ultra["cfd_flag_stoptn"].values[df["CFDStopTN"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStopBN"].values[df["CFDStopBN"].values != -1],
        decom_ultra["cfd_flag_stopbn"].values[df["CFDStopBN"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStopTE"].values[df["CFDStopTE"].values != -1],
        decom_ultra["cfd_flag_stopte"].values[df["CFDStopTE"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStopBE"].values[df["CFDStopBE"].values != -1],
        decom_ultra["cfd_flag_stopbe"].values[df["CFDStopBE"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStopTS"].values[df["CFDStopTS"].values != -1],
        decom_ultra["cfd_flag_stopts"].values[df["CFDStopTS"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStopBS"].values[df["CFDStopBS"].values != -1],
        decom_ultra["cfd_flag_stopbs"].values[df["CFDStopBS"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStopTW"].values[df["CFDStopTW"].values != -1],
        decom_ultra["cfd_flag_stoptw"].values[df["CFDStopTW"].values != -1],
    )
    np.testing.assert_array_equal(
        df["CFDStopBW"].values[df["CFDStopBW"].values != -1],
        decom_ultra["cfd_flag_stopbw"].values[df["CFDStopBW"].values != -1],
    )
