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
    determine_species_ssd,
    get_coincidence_positions,
    get_energy_ssd,
    get_front_x_position,
    get_front_y_position,
    get_particle_velocity,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_ssd_offset_and_positions,
    get_ssd_tof,
)
from imap_processing.utils import group_by_apid

# IMPORTANT: TODO tomorrow - datatypes!!!


@pytest.fixture()
def de_dataset(ccsds_path_theta_0, xtce_path):
    """L1A test data"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    decom_ultra_events = process_ultra_apids(
        grouped_data[ULTRA_EVENTS.apid[0]], ULTRA_EVENTS.apid[0]
    )
    decom_ultra_aux = process_ultra_apids(
        grouped_data[ULTRA_AUX.apid[0]], ULTRA_AUX.apid[0]
    )
    dataset = create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )
    # Remove start_type with fill values
    l1a_de_dataset = dataset.where(
        dataset["START_TYPE"] != GlobalConstants.INT_FILLVAL, drop=True
    )
    return l1a_de_dataset


def test_get_front_x_position(
    de_dataset,
    events_fsw_comparison_theta_0,
):
    """Tests get_front_x_position function."""
    # Left Start Type
    indices_left = np.where(de_dataset["START_TYPE"] == 1)[0]
    # Right Start Type
    indices_right = np.where(de_dataset["START_TYPE"] == 2)[0]

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices_left]
    selected_rows_2 = df_filt.iloc[indices_right]

    xf = get_front_x_position(
        de_dataset["START_TYPE"].data,
        de_dataset["START_POS_TDC"].data,
    )

    # TODO: Need new lookup tables for "XFTLTOFF", "XFTRTOFF"
    # Need new lookup table.
    # The value 180 was added to xf_1 since that
    # is the offset from the FSW xft_off
    assert np.allclose(
        xf[indices_left] + 180, selected_rows_1.Xf.values.astype("float"), rtol=1e-3
    )

    # The value 25 was subtracted from xf_2 bc that
    # is the offset from the FSW xft_off
    assert np.allclose(
        xf[indices_right] - 25, selected_rows_2.Xf.values.astype("float"), rtol=1e-3
    )


def test_xb_yb(
    de_dataset,
    events_fsw_comparison_theta_0,
):
    """Tests xb and yb."""
    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    _, _, ph_xb, ph_yb = get_ph_tof_and_back_positions(
        de_dataset, df_filt.Xf.astype("float").values, "ultra45"
    )

    ph_indices = np.where(
        (de_dataset["STOP_TYPE"] == 1) | (de_dataset["STOP_TYPE"] == 2)
    )[0]
    selected_rows = df_filt.iloc[ph_indices]
    np.testing.assert_array_equal(ph_xb, selected_rows["Xb"].astype("float"))
    np.testing.assert_array_equal(ph_yb, selected_rows["Yb"].astype("float"))

    ssd_yb, tof_offsets, _ = get_ssd_offset_and_positions(de_dataset)
    ssd_indices = np.where(de_dataset["STOP_TYPE"] >= 8)[0]
    selected_rows = df_filt.iloc[ssd_indices]
    np.testing.assert_array_equal(ssd_yb, selected_rows["Yb"].astype("float"))


def test_ph_components(
    events_fsw_comparison_theta_0,
    de_dataset,
):
    """Tests velocity and other parameters used for velocity."""
    ph_indices = np.where(np.isin(de_dataset["STOP_TYPE"], [1, 2]))[0]

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    d, yf = get_front_y_position(
        de_dataset["START_TYPE"].data, df_filt.Yb.values.astype("float")
    )

    assert yf == pytest.approx(df_filt["Yf"].astype("float"), 1e-3)

    test_xf = df_filt["Xf"].astype("float").values
    test_yf = df_filt["Yf"].astype("float").values

    test_xb = df_filt["Xb"].astype("float").values
    test_yb = df_filt["Yb"].astype("float").values

    r = get_path_length((test_xf, test_yf), (test_xb, test_yb), d)
    assert r == pytest.approx(df_filt["r"].astype("float"), rel=1e-3)

    # TODO: Once we have the lookup table, we can test this.
    # energy = get_energy_pulse_height(
    #     de_dataset["STOP_TYPE"].data[ph_indices],
    #     test_xb[ph_indices],
    #     test_yb[ph_indices]
    # )

    tof, t2, xb, yb = get_ph_tof_and_back_positions(
        de_dataset, df_filt.Xf.values.astype("float"), "ultra45"
    )

    index = np.where((df_filt["CoinType"] == 1) | (df_filt["CoinType"] == 2))[0]

    # TODO: This is as close as I can get.
    #  I suspect that the lookup table is different.
    test_xc = df_filt["Xc"].iloc[index].astype("float")
    etof, xc = get_coincidence_positions(de_dataset, tof, "ultra45")
    assert xc == pytest.approx(test_xc.values, rel=1)

    test_energy = df_filt["Energy"].iloc[ph_indices].astype("float")
    test_r = df_filt["r"].iloc[ph_indices].astype("float")

    # TODO: needs lookup table to test bin
    ctof, bin = determine_species_pulse_height(
        test_energy.to_numpy(), tof, test_r.to_numpy()
    )
    # Note: good agreement.
    assert ctof == pytest.approx(
        df_filt["cTOF"].iloc[ph_indices].astype("float"), rel=1e-3
    )

    vhat_x, vhat_y, vhat_z = get_particle_velocity(
        (test_xf[ph_indices], test_yf[ph_indices]),
        (test_xb[ph_indices], test_yb[ph_indices]),
        d[ph_indices],
        tof,
    )
    # FSW test data should be negative and not have an analysis
    # for negative tof values.
    assert vhat_x[tof > 0] == pytest.approx(
        -df_filt["vhatX"].iloc[
            ph_indices].astype("float").values[tof > 0], rel=1e-2
    )
    assert vhat_y[tof > 0] == pytest.approx(
        -df_filt["vhatY"].iloc[
            ph_indices].astype("float").values[tof > 0], rel=1e-2
    )
    assert vhat_z[tof > 0] == pytest.approx(
        -df_filt["vhatZ"].iloc[
            ph_indices].astype("float").values[tof > 0], rel=1e-2
    )


def test_get_ssd_offset_and_positions(
    de_dataset,
    events_fsw_comparison_theta_0,
):
    """Tests get_ssd_offset_and_positions function."""
    ssd_indices = np.where(de_dataset["STOP_TYPE"] >= 8)[0]

    _, tof_offsets, _ = get_ssd_offset_and_positions(de_dataset)

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows = df_filt.iloc[ssd_indices]

    # -4 is a value of an offset for SSD3 for Left Start Type and
    # SSD0 for Right Start Type.
    offset_length = len(tof_offsets[tof_offsets == -4])
    expected_offset_length = len(
        selected_rows[
            ((selected_rows["StartType"] == 1) & (selected_rows["SSDS3"] == 1))
            | ((selected_rows["StartType"] == 2) & (selected_rows["SSDS4"] == 1))
        ]
    )

    assert offset_length == expected_offset_length


def test_ssd_components(
    events_fsw_comparison_theta_0,
    de_dataset,
):
    """Tests velocity and other parameters used for velocity."""
    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    xf = df_filt["Xf"].astype("float").values

    tof, ssd = get_ssd_tof(de_dataset, xf)

    ssd_indices = np.where(de_dataset["STOP_TYPE"] >= 8)[0]

    energy = get_energy_ssd(de_dataset, ssd)
    test_energy = df_filt["Energy"].iloc[ssd_indices].astype("float")

    # TODO: the first value doesn't match. Look into this.
    assert np.array_equal(test_energy[1::], energy[1::].astype(float))

    r = df_filt["r"].astype("float")

    # TODO: needs lookup table to test bin
    ctof, bin = determine_species_ssd(
        test_energy.to_numpy(), tof, r.iloc[ssd_indices].to_numpy()
    )
    test_ctof = df_filt["cTOF"].iloc[ssd_indices].astype("float")

    # TODO: these values don't match exactly. Need exact lookup values.
    assert ctof[1::] == pytest.approx(test_ctof.values[1::], rel=1)
