"""Tests Extended Raw Events for ULTRA L1b."""

import pandas as pd
import pytest

from imap_processing.ultra.l1b.ultra_l1b_extended import (
    get_front_x_position,
    get_front_y_position,
)


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

    assert xf == pytest.approx(df_filt["Xf"].astype("float"), 1e-3)


def test_get_front_y_position(
    de_dataset,
    events_fsw_comparison_theta_0,
):
    """Tests get_front_y_position function."""

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    d, yf = get_front_y_position(
        de_dataset["START_TYPE"].data, df_filt.Yb.values.astype("float")
    )

    assert yf == pytest.approx(df_filt["Yf"].astype("float"), abs=1e-3)
