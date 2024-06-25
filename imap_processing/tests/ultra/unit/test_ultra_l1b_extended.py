# TODO: test get_energy_pulse_height
import numpy as np
import pandas as pd
import pytest

from imap_processing import decom
from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.ultra.l0.decom_ultra import process_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
)
from imap_processing.ultra.l1a.ultra_l1a import create_dataset
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    determine_species_pulse_height,
    get_back_positions,
    get_front_x_position,
    get_front_y_position,
    get_particle_velocity,
    get_path_length,
    get_ssd_positions,
    get_ssd_index,
)
from imap_processing.utils import group_by_apid


@pytest.fixture()
def decom_ultra_aux(ccsds_path_theta_0, xtce_path):
    """Data for decom_ultra_aux"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)

    data_packet_list = process_ultra_apids(
        grouped_data[ULTRA_AUX.apid[0]], ULTRA_AUX.apid[0]
    )
    return data_packet_list


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_EVENTS.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            },
        )
    ],
    indirect=True,
)
def test_get_front_x_position(
    decom_test_data,
    decom_ultra_aux,
    events_fsw_comparison_theta_0,
):
    """Tests get_front_x_position function."""
    decom_ultra_events, _ = decom_test_data
    dataset = create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )

    # Remove start_type with fill values
    events_dataset = dataset.where(
        dataset["START_TYPE"] != GlobalConstants.INT_FILLVAL, drop=True
    )

    # Check top and bottom
    indices_1 = np.where(events_dataset["START_TYPE"] == 1)[0]
    indices_2 = np.where(events_dataset["START_TYPE"] == 2)[0]

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices_1]
    selected_rows_2 = df_filt.iloc[indices_2]

    xf_1 = get_front_x_position(
        events_dataset["START_TYPE"].data[indices_1],
        events_dataset["START_POS_TDC"].data[indices_1],
    )
    xf_2 = get_front_x_position(
        events_dataset["START_TYPE"].data[indices_2],
        events_dataset["START_POS_TDC"].data[indices_2],
    )

    # The value 180 was added to xf_1 since that is the offset from the FSW xft_off
    assert np.allclose(xf_1 + 180, selected_rows_1.Xf.values.astype("float"), rtol=1e-3)
    # The value 25 was subtracted from xf_2 bc that is the offset from the FSW xft_off
    assert np.allclose(xf_2 - 25, selected_rows_2.Xf.values.astype("float"), rtol=1e-3)


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_EVENTS.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            },
        )
    ],
    indirect=True,
)
def test_xb_yb(
    decom_test_data,
    decom_ultra_aux,
    events_fsw_comparison_theta_0,
):
    """Tests xb and yb from get_back_positions function."""
    decom_ultra_events, _ = decom_test_data
    dataset = create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )

    # Remove start_type with fill values
    events_dataset = dataset.where(
        dataset["START_TYPE"] != GlobalConstants.INT_FILLVAL, drop=True
    )

    # Check top and bottom
    indices_1 = np.where(events_dataset["STOP_TYPE"] == 1)[0]
    indices_2 = np.where(events_dataset["STOP_TYPE"] == 2)[0]

    indices = np.concatenate((indices_1, indices_2))
    indices.sort()

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices]

    _, _, xb, yb = get_back_positions(
        indices, events_dataset, selected_rows_1.Xf.values.astype("float")
    )

    np.testing.assert_array_equal(xb[indices], selected_rows_1["Xb"].astype("float"))
    np.testing.assert_array_equal(yb[indices], selected_rows_1["Yb"].astype("float"))


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_EVENTS.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            },
        )
    ],
    indirect=True,
)
def test_yb_ssd(
    decom_test_data,
    decom_ultra_aux,
    events_fsw_comparison_theta_0,
):
    """Tests yb from get_ssd_index function."""
    decom_ultra_events, _ = decom_test_data
    dataset = create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )

    # Remove start_type with fill values
    events_dataset = dataset.where(
        dataset["START_TYPE"] != GlobalConstants.INT_FILLVAL, drop=True
    )

    indices = np.where(events_dataset["STOP_TYPE"] >= 8)[0]

    ssd_indices, ybs, _ = get_ssd_index(indices, events_dataset, "LT")

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows = df_filt.iloc[ssd_indices]

    np.testing.assert_array_equal(ybs, selected_rows["Yb"].astype("float"))


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_EVENTS.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            },
        )
    ],
    indirect=True,
)
def test_yf(
    events_fsw_comparison_theta_0,
    decom_test_data,
    decom_ultra_aux,
):
    decom_ultra_events, _ = decom_test_data
    dataset = create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )

    # Remove start_type with fill values
    events_dataset = dataset.where(
        dataset["START_TYPE"] != GlobalConstants.INT_FILLVAL, drop=True
    )

    indices_1 = np.where(events_dataset["STOP_TYPE"] == 1)[0]
    indices_2 = np.where(events_dataset["STOP_TYPE"] == 2)[0]
    indices = np.concatenate((indices_1, indices_2))
    indices.sort()

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices]

    d, yf = get_front_y_position(events_dataset, df_filt.Yb.values.astype("float"))

    assert yf == pytest.approx(df_filt["Yf"].astype("float"), 1e-3)

    xf_test = df_filt["Xf"].astype("float").values
    yf_test = df_filt["Yf"].astype("float").values

    xb_test = df_filt["Xb"].astype("float").values
    yb_test = df_filt["Yb"].astype("float").values

    r = get_path_length((xf_test, yf_test), (xb_test, yb_test), d)
    assert r == pytest.approx(df_filt["r"].astype("float"), rel=1e-3)

    # TODO: test get_energy_pulse_height
    # pulse_height = events_dataset["ENERGY_PH"].data[index]
    # energy = get_energy_pulse_height(pulse_height, stop_type, xb, yb)

    # TODO: needs lookup table to test bin
    tof, t2, xb, yb = get_back_positions(
        indices, events_dataset, selected_rows_1.Xf.values.astype("float")
    )

    energy = df_filt["Xf"].iloc[indices].astype("float")
    r = df_filt["r"].iloc[indices].astype("float")

    ctof, bin = determine_species_pulse_height(energy, tof[indices] * 100, r)
    assert ctof.values == pytest.approx(
        df_filt["cTOF"].iloc[indices].astype("float").values, rel=1e-3
    )

    vhat_x, vhat_y, vhat_z = get_particle_velocity(
        (xf_test[indices], yf_test[indices]),
        (xb_test[indices], yb_test[indices]),
        d[indices],
        tof[indices],
    )

    assert vhat_x == pytest.approx(
        df_filt["vhatX"].iloc[indices].astype("float").values, rel=1e-2
    )
    assert vhat_y == pytest.approx(
        df_filt["vhatY"].iloc[indices].astype("float").values, rel=1e-2
    )
    assert vhat_z == pytest.approx(
        df_filt["vhatZ"].iloc[indices].astype("float").values, rel=1e-2
    )
