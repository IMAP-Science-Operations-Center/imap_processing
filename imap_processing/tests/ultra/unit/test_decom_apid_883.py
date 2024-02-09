import pandas as pd
import pytest

from imap_processing.ultra.l0.decom_ultra import decom_image_ena_phxtof_hi_ang_packets


@pytest.fixture()
def decom_ultra(ccsds_path_image_ena_phxtof_hi_ang, xtce_image_ena_phxtof_hi_ang_path):
    """Data for decom_ultra"""
    data_packet_list = decom_image_ena_phxtof_hi_ang_packets(
        ccsds_path_image_ena_phxtof_hi_ang, xtce_image_ena_phxtof_hi_ang_path
    )
    return data_packet_list


def test_image_rate_decom(decom_ultra, image_ena_phxtof_hi_ang_test_path):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""

    df = pd.read_csv(image_ena_phxtof_hi_ang_test_path, index_col="MET")

    assert (df["SID"].values == decom_ultra["science_id"].values).all()
    assert (df["Spin"].values == decom_ultra["spin_data"].values).all()
    assert (df["AbortFlag"].values == decom_ultra["abortflag_data"].values).all()
    assert (df["StartDelay"].values == decom_ultra["startdelay_data"].values).all()
    assert (df["Count"].values == decom_ultra["count_data"].values).all()
    assert (df["CoinType"].values == decom_ultra["coin_type"].values).all()
    assert (df["StartType"].values == decom_ultra["start_type"].values).all()
    assert (df["StopType"].values == decom_ultra["stop_type"].values).all()
    assert (df["StartPosTDC"].values == decom_ultra["start_pos_tdc"].values).all()
    assert (df["StopNorthTDC"].values == decom_ultra["stop_north_tdc"].values).all()
    assert (df["StopEastTDC"].values == decom_ultra["stop_east_tdc"].values).all()
    assert (df["StopSouthTDC"].values == decom_ultra["stop_south_tdc"].values).all()
    assert (df["StopWestTDC"].values == decom_ultra["stop_west_tdc"].values).all()
    assert (df["CoinNorthTDC"].values == decom_ultra["coin_north_tdc"].values).all()
    assert (df["CoinSouthTDC"].values == decom_ultra["coin_south_tdc"].values).all()
    assert (
        df["CoinDiscreteTDC"].values == decom_ultra["coin_discrete_tdc"].values
    ).all()
    assert (df["EnergyOrPH"].values == decom_ultra["energy_ph"].values).all()
    assert (df["PulseWidth"].values == decom_ultra["pulse_width"].values).all()
