import pandas as pd
import pytest

from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.ultra.l0.ultra_utils import UltraParams


@pytest.fixture()
def decom_ultra(ccsds_path_image_raw_events, xtce_image_raw_events_path):
    """Data for decom_ultra"""
    data_packet_list = decom_ultra_apids(
        ccsds_path_image_raw_events, xtce_image_raw_events_path,
        UltraParams.ULTRA_EVENTS.value.apid[0]
    )
    return data_packet_list


def test_image_raw_events_decom(decom_ultra, image_raw_events_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(image_raw_events_test_path, index_col="MET")
    df.replace(-1, GlobalConstants.INT_FILLVAL, inplace=True)

    assert (df.SID == decom_ultra["SID"]).all()
    assert (df["Spin"] == decom_ultra["SPIN"]).all()
    assert (df["AbortFlag"] == decom_ultra["ABORTFLAG"]).all()
    assert (df["StartDelay"] == decom_ultra["STARTDELAY"]).all()
    assert (df["Count"] == decom_ultra["COUNT"]).all()
    assert (df["CoinType"] == decom_ultra["coin_type"]).all()
    assert (df["StartType"] == decom_ultra["start_type"]).all()
    assert (df["StopType"] == decom_ultra["stop_type"]).all()
    assert (df["StartPosTDC"] == decom_ultra["start_pos_tdc"]).all()
    assert (df["StopNorthTDC"] == decom_ultra["stop_north_tdc"]).all()
    assert (df["StopEastTDC"] == decom_ultra["stop_east_tdc"]).all()
    assert (df["StopSouthTDC"] == decom_ultra["stop_south_tdc"]).all()
    assert (df["StopWestTDC"] == decom_ultra["stop_west_tdc"]).all()
    assert (df["CoinNorthTDC"] == decom_ultra["coin_north_tdc"]).all()
    assert (df["CoinSouthTDC"] == decom_ultra["coin_south_tdc"]).all()
    assert (
        df["CoinDiscreteTDC"] == decom_ultra["coin_discrete_tdc"]
    ).all()
    assert (df["EnergyOrPH"] == decom_ultra["energy_ph"]).all()
    assert (df["PulseWidth"] == decom_ultra["pulse_width"]).all()
