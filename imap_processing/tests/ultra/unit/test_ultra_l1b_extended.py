"""Tests Extended Raw Events for ULTRA L1b."""

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l1b.ultra_l1b_extended import (
    StartType,
    StopType,
    CoinType,
    get_coincidence_positions,
    get_front_x_position,
    get_front_y_position,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_ssd_back_position_and_tof_offset,
)


@pytest.fixture()
def yf_fixture(de_dataset, events_fsw_comparison_theta_0):
    """Fixture to compute and return yf and related data."""
    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    d, yf = get_front_y_position(
        de_dataset["START_TYPE"].data, df_filt.Yb.values.astype("float")
    )

    return df_filt, d, yf


def test_get_front_x_position(
    de_dataset,
    yf_fixture,
):
    """Tests get_front_x_position function."""

    df_filt, _, _ = yf_fixture

    xf = get_front_x_position(
        de_dataset["START_TYPE"].data,
        de_dataset["START_POS_TDC"].data,
    )

    assert xf == pytest.approx(df_filt["Xf"].astype("float"), 1e-5)


def test_get_front_y_position(yf_fixture):
    """Tests get_front_y_position function."""
    df_filt, d, yf = yf_fixture

    assert yf == pytest.approx(df_filt["Yf"].astype("float"), abs=1e-5)
    assert d == pytest.approx(df_filt["d"].astype("float"), abs=1e-5)


def test_get_path_length(de_dataset, yf_fixture):
    """Tests get_path_length function."""

    df_filt, d, yf = yf_fixture

    test_xf = df_filt["Xf"].astype("float").values
    test_yf = df_filt["Yf"].astype("float").values

    test_xb = df_filt["Xb"].astype("float").values
    test_yb = df_filt["Yb"].astype("float").values
    r = get_path_length((test_xf, test_yf), (test_xb, test_yb), d)
    assert r == pytest.approx(df_filt["r"].astype("float"), abs=1e-5)


def test_get_ph_tof_and_back_positions(
    de_dataset,
    yf_fixture,
):
    """Tests get_ph_tof_and_back_positions function."""

    df_filt, _, _ = yf_fixture

    ph_tof, _, ph_xb, ph_yb = get_ph_tof_and_back_positions(
        de_dataset, df_filt.Xf.astype("float").values, "ultra45"
    )

    ph_indices = np.nonzero(
        np.isin(de_dataset["STOP_TYPE"], [StopType.Top.value, StopType.Bottom.value])
    )[0]

    selected_rows = df_filt.iloc[ph_indices]

    np.testing.assert_array_equal(ph_xb, selected_rows["Xb"].astype("float"))
    np.testing.assert_array_equal(ph_yb, selected_rows["Yb"].astype("float"))
    np.testing.assert_allclose(ph_tof, selected_rows["TOF"].astype("float"), atol=1e-5, rtol=0)


def test_get_ssd_back_position_and_tof_offset(
    de_dataset,
    events_fsw_comparison_theta_0,
):
    """Tests get_ssd_back_position function."""
    yb, tof_offset, ssd_number = get_ssd_back_position_and_tof_offset(de_dataset)

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[(df["StartType"] != -1) & (df["StopType"] >= 8)]

    np.testing.assert_array_equal(yb, df_filt["Yb"].astype("float"))

    tof_offset_lt = tof_offset[df_filt["StartType"] == StartType.Left.value]
    tof_offset_rt = tof_offset[df_filt["StartType"] == StartType.Right.value]

    ssd_number_lt = ssd_number[df_filt["StartType"] == StartType.Left.value]
    ssd_number_rt = ssd_number[df_filt["StartType"] == StartType.Right.value]

    np.testing.assert_array_equal(
        tof_offset_lt[ssd_number_lt == 3],
        np.full(len(tof_offset_lt[ssd_number_lt == 3]), -4.2),
    )
    np.testing.assert_array_equal(
        tof_offset_rt[ssd_number_rt == 7],
        np.full(len(tof_offset_rt[ssd_number_rt == 7]), -6),
    )
    np.testing.assert_array_equal(
        tof_offset_rt[ssd_number_rt == 4],
        np.full(len(tof_offset_rt[ssd_number_rt == 4]), -4),
    )

    assert np.all(ssd_number_lt >= 0), "Values in ssd_number_lt out of range."

    assert np.all(ssd_number_lt <= 7), "Values in ssd_number_lt out of range."

    assert np.all(ssd_number_rt >= 0), "Values in ssd_number_rt out of range."

    assert np.all(ssd_number_rt <= 7), "Values in ssd_number_rt out of range."


def test_get_coincidence_positions(de_dataset, yf_fixture):
    """Tests get_coincidence_positions function."""
    df_filt, _, _ = yf_fixture
    tof = df_filt["TOF"].astype("float").values

    etof, xc = get_coincidence_positions(de_dataset, tof, "ultra45")

    coin_indices = np.nonzero(
        np.isin(de_dataset["COIN_TYPE"], [CoinType.Top.value, CoinType.Bottom.value])
    )[0]

    coin_rows = df_filt.iloc[coin_indices]

    np.testing.assert_allclose(xc[coin_indices],
                               coin_rows["Xc"].astype("float"), atol=1e-4, rtol=0)

    # look at etof
    coin_top_indices = np.nonzero(
        np.isin(de_dataset["COIN_TYPE"], [CoinType.Top.value])
    )[0]
    coin_bottom_indices = np.nonzero(
        np.isin(de_dataset["COIN_TYPE"], [CoinType.Bottom.value])
    )[0]

    top_rows = df_filt.iloc[coin_top_indices]
    bottom_rows = df_filt.iloc[coin_bottom_indices]

    np.testing.assert_allclose(etof[coin_bottom_indices],
                               bottom_rows["eTOF"].astype("float"))

    # assert top
    np.testing.assert_allclose(etof[coin_top_indices],
                               top_rows["eTOF"].astype("float"))


    print('hi')
