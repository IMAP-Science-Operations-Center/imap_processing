"""Tests Extended Raw Events for ULTRA L1b."""

import pandas as pd
import pytest

from imap_processing.ultra.l1b.ultra_l1b_extended import (
    get_front_x_position,
    get_front_y_position,
    get_path_length,
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
    events_fsw_comparison_theta_0,
):
    """Tests get_front_x_position function."""

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    xf = get_front_x_position(
        de_dataset["START_TYPE"].data,
        de_dataset["START_POS_TDC"].data,
    )

    assert xf == pytest.approx(df_filt["Xf"].astype("float"), 1e-5)


def test_get_front_y_position(yf_fixture):
    """Tests get_front_y_position function."""
    df_filt, d, yf = yf_fixture

    assert yf == pytest.approx(df_filt["Yf"].astype("float"), abs=1e-5)


def test_get_path_length(de_dataset, yf_fixture):
    """Tests get_path_length function."""

    df_filt, d, yf = yf_fixture

    test_xf = df_filt["Xf"].astype("float").values
    test_yf = df_filt["Yf"].astype("float").values

    test_xb = df_filt["Xb"].astype("float").values
    test_yb = df_filt["Yb"].astype("float").values
    r = get_path_length((test_xf, test_yf), (test_xb, test_yb), d)
    assert r == pytest.approx(df_filt["r"].astype("float"), abs=1e-5)
