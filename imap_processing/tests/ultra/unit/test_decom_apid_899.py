import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l0.decom_ultra import process_ultra_events
from imap_processing.ultra.l0.ultra_utils import ULTRA_PRI_2_EVENTS
from imap_processing.utils import packet_file_to_datasets


@pytest.mark.external_test_data
def test_image_raw_events_decom(xtce_path, priority_2_test_path, ccsds_path_extra):
    """This function reads validation data and checks that decom data
    matches validation data for image rate packet"""
    datasets_by_apid = packet_file_to_datasets(ccsds_path_extra, xtce_path)
    decom_ultra = process_ultra_events(
        datasets_by_apid[ULTRA_PRI_2_EVENTS.apid[0]], ULTRA_PRI_2_EVENTS.apid[0]
    )
    df = pd.read_csv(priority_2_test_path, index_col="MET")

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
            df[df_var].values[good_values],
            decom_ultra[xr_var].values[good_values],
        )
