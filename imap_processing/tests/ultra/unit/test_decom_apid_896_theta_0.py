import numpy as np
import pandas as pd
import pytest

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import ULTRA_EVENTS


@pytest.fixture()
def decom_ultra(ccsds_path_theta_0, xtce_path):
    """Data for decom_ultra"""
    data_packet_list = decom_ultra_apids(
        ccsds_path_theta_0, xtce_path, ULTRA_EVENTS.apid[0]
    )
    # Convert dictionary to DataFrame
    decom_ultra_df = pd.DataFrame(data_packet_list)

    # Now apply the filtering
    filtered_decom_ultra = decom_ultra_df[
        (decom_ultra_df["COIN_TYPE"] != GlobalConstants.INT_FILLVAL)
    ]
    return filtered_decom_ultra


def test_image_raw_events_decom(decom_ultra, events_test_path_theta_0):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(events_test_path_theta_0, index_col="TimeTag")

    np.testing.assert_array_equal(df["Spin"], decom_ultra["SPIN"])
    np.testing.assert_array_equal(df["AbortFlag"], decom_ultra["ABORTFLAG"])
    np.testing.assert_array_equal(df["StartDelay"], decom_ultra["STARTDELAY"])
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
    np.testing.assert_array_equal(df["CoinDiscrete"], decom_ultra["COIN_DISCRETE_TDC"])
    np.testing.assert_array_equal(df["EnergyPH"], decom_ultra["ENERGY_PH"])
    np.testing.assert_array_equal(df["PulseWidth"], decom_ultra["PULSE_WIDTH"])
    np.testing.assert_array_equal(df["Bin"], decom_ultra["BIN"])


def test_image_raw_events_decom_flags(decom_ultra, events_test_path_theta_0):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(events_test_path_theta_0, index_col="TimeTag")

    np.testing.assert_array_equal(df["PHCompSL"], decom_ultra["EVENT_FLAG_PHCMPSL"])
    np.testing.assert_array_equal(df["PHCompSR"], decom_ultra["EVENT_FLAG_PHCMPSR"])
    np.testing.assert_array_equal(df["PHCompCD"], decom_ultra["EVENT_FLAG_PHCMPCD"])
