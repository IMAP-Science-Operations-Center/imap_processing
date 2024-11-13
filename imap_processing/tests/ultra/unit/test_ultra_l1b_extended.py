"""Tests Extended Raw Events for ULTRA L1b."""

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l1b.ultra_l1b_extended import (
    CoinType,
    StartType,
    StopType,
    calculate_etof_xc,
    determine_species_pulse_height,
    determine_species_ssd,
    get_coincidence_positions,
    get_ctof,
    get_energy_pulse_height,
    get_energy_ssd,
    get_front_x_position,
    get_front_y_position,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_ssd_back_position_and_tof_offset,
    get_ssd_tof,
    get_unit_vector,
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
    np.testing.assert_allclose(
        ph_tof, selected_rows["TOF"].astype("float"), atol=1e-5, rtol=0
    )


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
    # Get particle tof (t2).
    _, t2, _, _ = get_ph_tof_and_back_positions(
        de_dataset, df_filt.Xf.astype("float").values, "ultra45"
    )

    # Filter for stop type.
    indices = np.nonzero(
        np.isin(de_dataset["STOP_TYPE"], [StopType.Top.value, StopType.Bottom.value])
    )[0]
    de_filtered = de_dataset.isel(epoch=indices)
    rows = df_filt.iloc[indices]

    # Get coincidence position and eTOF.
    etof, xc = get_coincidence_positions(de_filtered, t2, "ultra45")

    np.testing.assert_allclose(xc, rows["Xc"].astype("float"), atol=1e-4, rtol=0)
    np.testing.assert_allclose(
        etof, rows["eTOF"].astype("float").values, rtol=0, atol=1e-06
    )


def test_calculate_etof_xc(de_dataset, yf_fixture):
    """Tests calculate_etof_xc function."""
    df_filt, _, _ = yf_fixture
    # Get particle tof (t2).
    _, t2, _, _ = get_ph_tof_and_back_positions(
        de_dataset, df_filt.Xf.astype("float").values, "ultra45"
    )
    # Filter based on STOP_TYPE.
    indices = np.nonzero(
        np.isin(de_dataset["STOP_TYPE"], [StopType.Top.value, StopType.Bottom.value])
    )[0]
    de_filtered = de_dataset.isel(epoch=indices)
    df_filtered = df_filt.iloc[indices]

    # Filter for COIN_TYPE Top and Bottom.
    index_top = np.nonzero(np.isin(de_filtered["COIN_TYPE"], CoinType.Top.value))[0]
    de_top = de_filtered.isel(epoch=index_top)
    df_top = df_filtered.iloc[index_top]

    index_bottom = np.nonzero(np.isin(de_filtered["COIN_TYPE"], CoinType.Bottom.value))[
        0
    ]
    de_bottom = de_filtered.isel(epoch=index_bottom)
    df_bottom = df_filtered.iloc[index_bottom]

    # Calculate for Top and Bottom
    etof_top, xc_top = calculate_etof_xc(de_top, t2[index_top], "ultra45", "TP")
    etof_bottom, xc_bottom = calculate_etof_xc(
        de_bottom, t2[index_bottom], "ultra45", "BT"
    )

    # Assertions for Top
    np.testing.assert_allclose(
        xc_top * 100, df_top["Xc"].astype("float"), atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(
        etof_top, df_top["eTOF"].astype("float").values, atol=1e-06, rtol=0
    )

    # Assertions for Bottom
    np.testing.assert_allclose(
        xc_bottom * 100, df_bottom["Xc"].astype("float"), atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(
        etof_bottom, df_bottom["eTOF"].astype("float").values, atol=1e-06, rtol=0
    )


def test_get_unit_vector(de_dataset, yf_fixture):
    """Tests get_unit_vector function."""
    df_filt, _, _ = yf_fixture

    ph_indices = np.nonzero(
        np.isin(de_dataset["STOP_TYPE"], [StopType.Top.value, StopType.Bottom.value])
    )[0]

    ph_rows = df_filt.iloc[ph_indices]
    test_xf = ph_rows["Xf"].astype("float").values
    test_yf = ph_rows["Yf"].astype("float").values
    test_xb = ph_rows["Xb"].astype("float").values
    test_yb = ph_rows["Yb"].astype("float").values
    test_d = ph_rows["d"].astype("float").values
    test_tof = ph_rows["TOF"].astype("float").values

    vhat_x, vhat_y, vhat_z = get_unit_vector(
        (test_xf, test_yf),
        (test_xb, test_yb),
        test_d,
        test_tof,
    )
    # FSW test data should be negative and not have an analysis
    # for negative tof values.
    assert vhat_x[test_tof > 0] == pytest.approx(
        -df_filt["vhatX"].iloc[ph_indices].astype("float").values[test_tof > 0],
        rel=1e-2,
    )
    assert vhat_y[test_tof > 0] == pytest.approx(
        -df_filt["vhatY"].iloc[ph_indices].astype("float").values[test_tof > 0],
        rel=1e-2,
    )
    assert vhat_z[test_tof > 0] == pytest.approx(
        -df_filt["vhatZ"].iloc[ph_indices].astype("float").values[test_tof > 0],
        rel=1e-2,
    )


def test_get_ssd_tof(de_dataset, yf_fixture):
    """Tests get_ssd_tof function."""
    df_filt, _, _ = yf_fixture
    df_ssd = df_filt[df_filt["StopType"].isin(StopType.SSD.value)]
    test_xf = df_filt["Xf"].astype("float").values

    ssd_tof = get_ssd_tof(de_dataset, test_xf)

    np.testing.assert_allclose(
        ssd_tof, df_ssd["TOF"].astype("float"), atol=1e-05, rtol=0
    )


def test_get_energy_ssd(de_dataset, yf_fixture):
    """Tests get_energy_ssd function."""
    df_filt, _, _ = yf_fixture
    df_ssd = df_filt[df_filt["StopType"].isin(StopType.SSD.value)]
    _, _, ssd_number = get_ssd_back_position_and_tof_offset(de_dataset)
    energy = get_energy_ssd(de_dataset, ssd_number)
    test_energy = df_ssd["Energy"].astype("float")

    assert np.array_equal(test_energy, energy)


def test_get_energy_pulse_height(de_dataset, yf_fixture):
    """Tests get_energy_ssd function."""
    df_filt, _, _ = yf_fixture
    df_ph = df_filt[df_filt["StopType"].isin(StopType.PH.value)]
    ph_indices = np.nonzero(
        np.isin(de_dataset["STOP_TYPE"], [StopType.Top.value, StopType.Bottom.value])
    )[0]

    test_xb = df_filt["Xb"].astype("float").values
    test_yb = df_filt["Yb"].astype("float").values

    energy = get_energy_pulse_height(
        de_dataset["STOP_TYPE"].data, de_dataset["ENERGY_PH"].data, test_xb, test_yb
    )
    test_energy = df_ph["Energy"].astype("float")

    assert np.array_equal(test_energy, energy[ph_indices])


def test_get_ctof(yf_fixture):
    """Tests get_ctof function."""
    df_filt, _, _ = yf_fixture

    df_ph = df_filt[df_filt["StopType"].isin([StopType.PH.value])]

    df_ssd = df_filt[df_filt["StopType"].isin([StopType.SSD.value])]

    ph_ctof = get_ctof(
        df_ph["TOF"].astype("float").to_numpy(),
        df_ph["r"].astype("float").to_numpy(),
        "PH",
    )

    ssd_ctof = get_ctof(
        df_ssd["TOF"].astype("float").to_numpy(),
        df_ssd["r"].astype("float").to_numpy(),
        "SSD",
    )

    np.testing.assert_allclose(
        ph_ctof, df_ph["cTOF"].astype("float"), atol=1e-05, rtol=0
    )
    np.testing.assert_allclose(
        ssd_ctof, df_ssd["cTOF"].astype("float"), atol=1e-05, rtol=0
    )


def test_determine_species_ph(yf_fixture):
    """Tests determine_species_ph function."""
    df_filt, _, _ = yf_fixture
    df_ph = df_filt[df_filt["StopType"].isin(StopType.PH.value)]

    bin = determine_species_pulse_height(
        df_ph["Energy"].astype("float").to_numpy(),
        df_ph["TOF"].astype("float").to_numpy(),
        df_ph["r"].astype("float").to_numpy(),
    )

    # TODO: add in bin values.
    np.testing.assert_allclose(bin, np.zeros(len(bin)), atol=1e-05, rtol=0)


def test_determine_species_ssd(yf_fixture):
    """Tests determine_species_ssd function."""
    df_filt, _, _ = yf_fixture
    df_ssd = df_filt[df_filt["StopType"].isin(StopType.SSD.value)]

    bin = determine_species_ssd(
        df_ssd["Energy"].astype("float").to_numpy(),
        df_ssd["TOF"].astype("float").to_numpy(),
        df_ssd["r"].astype("float").to_numpy(),
    )

    # TODO: add in bin values.
    np.testing.assert_allclose(bin, np.zeros(len(bin)), atol=1e-05, rtol=0)
