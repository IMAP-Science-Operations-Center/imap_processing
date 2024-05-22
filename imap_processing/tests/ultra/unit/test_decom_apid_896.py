import numpy as np
import pandas as pd
import pytest

from imap_processing import decom
from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import ULTRA_EVENTS
from imap_processing.utils import group_by_apid


@pytest.fixture()
def decom_ultra(ccsds_path_events, xtce_path):
    """Data for decom_ultra"""
    packets = decom.decom_packets(ccsds_path_events, xtce_path)
    grouped_data = group_by_apid(packets)
    data = {ULTRA_EVENTS.apid[0]: grouped_data[ULTRA_EVENTS.apid[0]]}

    data_packet_list = decom_ultra_apids(data, ULTRA_EVENTS.apid[0])
    return data_packet_list


def test_image_raw_events_decom(decom_ultra, events_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(events_test_path, index_col="MET")
    df.replace(-1, GlobalConstants.INT_FILLVAL, inplace=True)

    np.testing.assert_array_equal(df.SID, decom_ultra["SID"])
    np.testing.assert_array_equal(df["Spin"], decom_ultra["SPIN"])
    np.testing.assert_array_equal(df["AbortFlag"], decom_ultra["ABORTFLAG"])
    np.testing.assert_array_equal(df["StartDelay"], decom_ultra["STARTDELAY"])
    np.testing.assert_array_equal(df["Count"], decom_ultra["COUNT"])
    np.testing.assert_array_equal(df["CoinType"], decom_ultra["COIN_TYPE"])
    np.testing.assert_array_equal(df["StartType"], decom_ultra["START_TYPE"])
    np.testing.assert_array_equal(df["StopType"], decom_ultra["STOP_TYPE"])
    np.testing.assert_array_equal(df["StartPosTDC"], decom_ultra["START_POS_TDC"])
    np.testing.assert_array_equal(df["StopNorthTDC"], decom_ultra["STOP_NORTH_TDC"])
    np.testing.assert_array_equal(df["StopEastTDC"], decom_ultra["STOP_EAST_TDC"])
    np.testing.assert_array_equal(df["StopSouthTDC"], decom_ultra["STOP_SOUTH_TDC"])
    np.testing.assert_array_equal(df["StopWestTDC"], decom_ultra["STOP_WEST_TDC"])
    np.testing.assert_array_equal(df["CoinNorthTDC"], decom_ultra["COIN_NORTH_TDC"])
    np.testing.assert_array_equal(df["CoinSouthTDC"], decom_ultra["COIN_SOUTH_TDC"])
    np.testing.assert_array_equal(
        df["CoinDiscreteTDC"], decom_ultra["COIN_DISCRETE_TDC"]
    )
    np.testing.assert_array_equal(df["EnergyOrPH"], decom_ultra["ENERGY_PH"])
    np.testing.assert_array_equal(df["PulseWidth"], decom_ultra["PULSE_WIDTH"])
    np.testing.assert_array_equal(df["PhaseAngle"], decom_ultra["PHASE_ANGLE"])
    np.testing.assert_array_equal(df["Bin"], decom_ultra["BIN"])


def test_image_raw_events_decom_flags(decom_ultra, events_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(events_test_path, index_col="MET")
    df.replace(-1, GlobalConstants.INT_FILLVAL, inplace=True)

    np.testing.assert_array_equal(df["CnT"], decom_ultra["EVENT_FLAG_CNT"])
    np.testing.assert_array_equal(df["PHCmpSL"], decom_ultra["EVENT_FLAG_PHCMPSL"])
    np.testing.assert_array_equal(df["PHCmpSR"], decom_ultra["EVENT_FLAG_PHCMPSR"])
    np.testing.assert_array_equal(df["PHCmpCD"], decom_ultra["EVENT_FLAG_PHCMPCD"])
    np.testing.assert_array_equal(df["SSDS7"], decom_ultra["SSD_FLAG_7"])
    np.testing.assert_array_equal(df["SSDS6"], decom_ultra["SSD_FLAG_6"])
    np.testing.assert_array_equal(df["SSDS5"], decom_ultra["SSD_FLAG_5"])
    np.testing.assert_array_equal(df["SSDS4"], decom_ultra["SSD_FLAG_4"])
    np.testing.assert_array_equal(df["SSDS3"], decom_ultra["SSD_FLAG_3"])
    np.testing.assert_array_equal(df["SSDS2"], decom_ultra["SSD_FLAG_2"])
    np.testing.assert_array_equal(df["SSDS1"], decom_ultra["SSD_FLAG_1"])
    np.testing.assert_array_equal(df["SSDS0"], decom_ultra["SSD_FLAG_0"])
    np.testing.assert_array_equal(df["CFDCoinTN"], decom_ultra["CFD_FLAG_COINTN"])
    np.testing.assert_array_equal(df["CFDCoinBN"], decom_ultra["CFD_FLAG_COINBN"])
    np.testing.assert_array_equal(df["CFDCoinTS"], decom_ultra["CFD_FLAG_COINTS"])
    np.testing.assert_array_equal(df["CFDCoinBS"], decom_ultra["CFD_FLAG_COINBS"])
    np.testing.assert_array_equal(df["CFDCoinD"], decom_ultra["CFD_FLAG_COIND"])
    np.testing.assert_array_equal(df["CFDStartRF"], decom_ultra["CFD_FLAG_STARTRF"])
    np.testing.assert_array_equal(df["CFDStartLF"], decom_ultra["CFD_FLAG_STARTLF"])
    np.testing.assert_array_equal(df["CFDStartRP"], decom_ultra["CFD_FLAG_STARTRP"])
    np.testing.assert_array_equal(df["CFDStartLP"], decom_ultra["CFD_FLAG_STARTLP"])
    np.testing.assert_array_equal(df["CFDStopTN"], decom_ultra["CFD_FLAG_STOPTN"])
    np.testing.assert_array_equal(df["CFDStopBN"], decom_ultra["CFD_FLAG_STOPBN"])
    np.testing.assert_array_equal(df["CFDStopTE"], decom_ultra["CFD_FLAG_STOPTE"])
    np.testing.assert_array_equal(df["CFDStopBE"], decom_ultra["CFD_FLAG_STOPBE"])
    np.testing.assert_array_equal(df["CFDStopTS"], decom_ultra["CFD_FLAG_STOPTS"])
    np.testing.assert_array_equal(df["CFDStopBS"], decom_ultra["CFD_FLAG_STOPBS"])
    np.testing.assert_array_equal(df["CFDStopTW"], decom_ultra["CFD_FLAG_STOPTW"])
    np.testing.assert_array_equal(df["CFDStopBW"], decom_ultra["CFD_FLAG_STOPBW"])
