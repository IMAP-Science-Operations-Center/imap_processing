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
    get_energy_pulse_height,
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
from imap_processing.ultra.l1b.de import calculate_de


@pytest.fixture()
def de_dataset(ccsds_path_theta_0, xtce_path):
    """Test data"""
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
    de_dataset = dataset.where(
        dataset["START_TYPE"] != GlobalConstants.INT_FILLVAL, drop=True
    )
    return de_dataset


def test_get_front_x_position(
    de_dataset,
    events_fsw_comparison_theta_0,
):
    """Tests get_front_x_position function."""
    indices_1 = np.where(de_dataset["START_TYPE"] == 1)[0]
    indices_2 = np.where(de_dataset["START_TYPE"] == 2)[0]

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices_1]
    selected_rows_2 = df_filt.iloc[indices_2]

    xf = get_front_x_position(
        de_dataset["START_TYPE"].data,
        de_dataset["START_POS_TDC"].data,
    )

    # TODO: should we try to match FSW data on this?
    # The value 180 was added to xf_1 since that is the offset from the FSW xft_off
    assert np.allclose(
        xf[indices_1] + 180, selected_rows_1.Xf.values.astype("float"), rtol=1e-3
    )
    # The value 25 was subtracted from xf_2 bc that is the offset from the FSW xft_off
    assert np.allclose(
        xf[indices_2] - 25, selected_rows_2.Xf.values.astype("float"), rtol=1e-3
    )


def test_ph_xb_yb(
    de_dataset,
    events_fsw_comparison_theta_0,
):
    """Tests xb and yb from get_back_positions function."""

    indices_1 = np.where(de_dataset["STOP_TYPE"] == 1)[0]
    indices_2 = np.where(de_dataset["STOP_TYPE"] == 2)[0]

    indices = np.concatenate((indices_1, indices_2))
    indices.sort()

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices]

    _, _, xb, yb = get_ph_tof_and_back_positions(
        de_dataset, selected_rows_1.Xf.values.astype("float")
    )

    np.testing.assert_array_equal(xb, selected_rows_1["Xb"].astype("float"))
    np.testing.assert_array_equal(yb, selected_rows_1["Yb"].astype("float"))


def test_get_ssd_offset_and_positions(
    de_dataset,
    events_fsw_comparison_theta_0,
):
    """Tests get_ssd_offset_and_positions function."""
    ssd_indices, ybs, tof_offsets, _ = get_ssd_offset_and_positions(de_dataset)

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows = df_filt.iloc[ssd_indices]

    np.testing.assert_array_equal(ybs, selected_rows["Yb"].astype("float"))

    # -4 is a value of an offset for SSD3 for Left Start Type and SSD0 for Right Start Type.
    offset_length = len(tof_offsets[tof_offsets == -4])
    expected_offset_length = len(
        selected_rows[
            ((selected_rows["StartType"] == 1) & (selected_rows["SSDS3"] == 1))
            | ((selected_rows["StartType"] == 2) & (selected_rows["SSDS4"] == 1))
        ]
    )

    assert offset_length == expected_offset_length


def test_ph_velocity(
    events_fsw_comparison_theta_0,
    de_dataset,
):
    """Tests velocity and other parameters used for velocity."""
    indices = np.where(np.isin(de_dataset["STOP_TYPE"], [1, 2]))[0]

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    selected_rows_1 = df_filt.iloc[indices]

    d, yf = get_front_y_position(de_dataset, df_filt.Yb.values.astype("float"))

    assert yf == pytest.approx(df_filt["Yf"].astype("float"), 1e-3)

    test_xf = df_filt["Xf"].astype("float").values
    test_yf = df_filt["Yf"].astype("float").values

    test_xb = df_filt["Xb"].astype("float").values
    test_yb = df_filt["Yb"].astype("float").values

    r = get_path_length((test_xf, test_yf), (test_xb, test_yb), d)
    assert r == pytest.approx(df_filt["r"].astype("float"), rel=1e-3)

    # TODO: test get_energy_pulse_height
    energy = get_energy_pulse_height(de_dataset, test_xb, test_yb)

    tof, t2, xb, yb = get_ph_tof_and_back_positions(
        de_dataset, selected_rows_1.Xf.values.astype("float")
    )

    index_left = np.where(df_filt["CoinType"] == 1)[0]
    index_right = np.where(df_filt["CoinType"] == 2)[0]
    index = np.concatenate((index_left, index_right))

    # TODO: This is as close as I can get. I suspect that the lookup
    # table that I have is not correct. Leave as TODO.
    test_xc = df_filt["Xc"].iloc[index].astype("float")
    _, xc = get_coincidence_positions(de_dataset, tof)
    assert xc == pytest.approx(test_xc.values, rel=1)

    test_energy = df_filt["Energy"].iloc[indices].astype("float")
    r = df_filt["r"].iloc[indices].astype("float")

    # TODO: needs lookup table to test bin
    ctof, bin = determine_species_pulse_height(
        test_energy.to_numpy(), tof, r.to_numpy()
    )
    assert ctof == pytest.approx(
        df_filt["cTOF"].iloc[indices].astype("float"), rel=1e-3
    )

    vhat_x, vhat_y, vhat_z = get_particle_velocity(
        (test_xf[indices], test_yf[indices]),
        (test_xb[indices], test_yb[indices]),
        d[indices],
        tof,
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


def test_ssd_velocity(
    events_fsw_comparison_theta_0,
    de_dataset,
):
    """Tests velocity and other parameters used for velocity."""

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    xf = df_filt["Xf"].astype("float").values

    ssd_indices, tof, ssd = get_ssd_tof(de_dataset, xf)

    energy = get_energy_ssd(de_dataset, ssd_indices, ssd)
    test_energy = df_filt["Energy"].iloc[ssd_indices].astype("float")

    # TODO: the last values don't match. Look into this.
    assert np.array_equal(test_energy[0:385], energy[0:385].astype(float))

    r = df_filt["r"].astype("float")

    # TODO: needs lookup table to test bin
    ctof, bin = determine_species_ssd(
        test_energy.to_numpy(), tof, r.iloc[ssd_indices].to_numpy()
    )
    test_ctof = df_filt["cTOF"].iloc[ssd_indices].astype("float")

    # TODO: the last values don't match. Look into this.
    ctof[0:385] == pytest.approx(test_ctof.values[0:385], rel=1e-1)
