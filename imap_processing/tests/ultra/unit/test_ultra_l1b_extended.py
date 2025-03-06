"""Tests Extended Raw Events for ULTRA L1b."""

import numpy as np
import pandas as pd
import pytest

from imap_processing.spice.spin import get_spin_data
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    CoinType,
    StartType,
    StopType,
    calculate_etof_xc,
    determine_species,
    get_coincidence_positions,
    get_ctof,
    get_de_energy_kev,
    get_de_velocity,
    get_energy_pulse_height,
    get_energy_ssd,
    get_eventtimes,
    get_front_x_position,
    get_front_y_position,
    get_path_length,
    get_ph_tof_and_back_positions,
    get_phi_theta,
    get_ssd_back_position_and_tof_offset,
    get_ssd_tof,
)


@pytest.fixture()
def test_fixture(de_dataset, events_fsw_comparison_theta_0):
    """Fixture to compute and return yf and related data."""
    # Remove start_type with fill values
    de_dataset = de_dataset.where(
        de_dataset["START_TYPE"] != np.iinfo(np.int64).min, drop=True
    )

    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]

    d, yf = get_front_y_position(
        de_dataset["START_TYPE"].data, df_filt.Yb.values.astype("float")
    )

    return df_filt, d, yf, de_dataset


def test_get_front_x_position(
    test_fixture,
):
    """Tests get_front_x_position function."""

    df_filt, _, _, de_dataset = test_fixture

    xf = get_front_x_position(
        de_dataset["START_TYPE"].data,
        de_dataset["START_POS_TDC"].data,
    )

    assert xf == pytest.approx(df_filt["Xf"].astype("float"), 1e-5)


def test_get_front_y_position(test_fixture):
    """Tests get_front_y_position function."""
    df_filt, d, yf, _ = test_fixture

    assert yf == pytest.approx(df_filt["Yf"].astype("float"), abs=1e-5)
    assert d == pytest.approx(df_filt["d"].astype("float"), abs=1e-5)


def test_get_path_length(test_fixture):
    """Tests get_path_length function."""

    df_filt, d, yf, _ = test_fixture

    test_xf = df_filt["Xf"].astype("float").values
    test_yf = df_filt["Yf"].astype("float").values

    test_xb = df_filt["Xb"].astype("float").values
    test_yb = df_filt["Yb"].astype("float").values
    r = get_path_length((test_xf, test_yf), (test_xb, test_yb), d)
    assert r == pytest.approx(df_filt["r"].astype("float"), abs=1e-5)


def test_get_ph_tof_and_back_positions(
    test_fixture,
):
    """Tests get_ph_tof_and_back_positions function."""

    df_filt, _, _, de_dataset = test_fixture

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
    test_fixture,
    events_fsw_comparison_theta_0,
):
    """Tests get_ssd_back_position function."""
    _, _, _, de_dataset = test_fixture
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


def test_get_coincidence_positions(test_fixture):
    """Tests get_coincidence_positions function."""
    df_filt, _, _, de_dataset = test_fixture
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


def test_calculate_etof_xc(test_fixture):
    """Tests calculate_etof_xc function."""
    df_filt, _, _, de_dataset = test_fixture
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


def test_get_de_velocity(test_fixture):
    """Tests get_de_velocity function."""
    df_filt, _, _, _ = test_fixture
    df_ph = df_filt[np.isin(df_filt["StopType"], [StopType.PH.value])]

    test_xf, test_yf, test_xb, test_yb, test_d, test_tof = (
        df_ph[col].astype("float").values
        for col in ["Xf", "Yf", "Xb", "Yb", "d", "TOF"]
    )

    v = get_de_velocity(
        (test_xf, test_yf),
        (test_xb, test_yb),
        test_d,
        test_tof,
    )

    v_x, v_y, v_z = v[:, 0], v[:, 1], v[:, 2]

    np.testing.assert_allclose(
        v_x[test_tof > 0],
        df_ph["vx"].astype("float").values[test_tof > 0],
        atol=1e-01,
        rtol=0,
    )
    np.testing.assert_allclose(
        v_y[test_tof > 0],
        df_ph["vy"].astype("float").values[test_tof > 0],
        atol=1e-01,
        rtol=0,
    )
    np.testing.assert_allclose(
        v_z[test_tof > 0],
        df_ph["vz"].astype("float").values[test_tof > 0],
        atol=1e-01,
        rtol=0,
    )


def test_get_ssd_tof(test_fixture):
    """Tests get_ssd_tof function."""
    df_filt, _, _, de_dataset = test_fixture
    df_ssd = df_filt[np.isin(df_filt["StopType"], [StopType.SSD.value])]
    test_xf = df_filt["Xf"].astype("float").values

    ssd_tof = get_ssd_tof(de_dataset, test_xf)

    np.testing.assert_allclose(
        ssd_tof, df_ssd["TOF"].astype("float"), atol=1e-05, rtol=0
    )


def test_get_de_energy_kev(test_fixture):
    """Tests get_de_energy_kev function."""
    df_filt, _, _, _ = test_fixture
    df_ph = df_filt[np.isin(df_filt["StopType"], [StopType.PH.value])]
    df_ph = df_ph[df_ph["energy_revised"].astype("str") != "FILL"]

    species_bin_ph = determine_species(
        df_ph["TOF"].astype("float").to_numpy(),
        df_ph["r"].astype("float").to_numpy(),
        "PH",
    )
    test_xf, test_yf, test_xb, test_yb, test_d, test_tof = (
        df_ph[col].astype("float").values
        for col in ["Xf", "Yf", "Xb", "Yb", "d", "TOF"]
    )

    v = get_de_velocity(
        (test_xf, test_yf),
        (test_xb, test_yb),
        test_d,
        test_tof,
    )

    energy = get_de_energy_kev(v, species_bin_ph)
    index_hydrogen = np.where(species_bin_ph == "H")
    actual_energy = energy[index_hydrogen[0]]
    expected_energy = df_ph["energy_revised"].astype("float")

    np.testing.assert_allclose(actual_energy, expected_energy, atol=1e-01, rtol=0)


def test_get_energy_ssd(test_fixture):
    """Tests get_energy_ssd function."""
    df_filt, _, _, de_dataset = test_fixture
    df_ssd = df_filt[np.isin(df_filt["StopType"], [StopType.SSD.value])]
    _, _, ssd_number = get_ssd_back_position_and_tof_offset(de_dataset)
    energy = get_energy_ssd(de_dataset, ssd_number)
    test_energy = df_ssd["Energy"].astype("float")

    assert np.array_equal(test_energy, energy)


def test_get_energy_pulse_height(test_fixture):
    """Tests get_energy_ssd function."""
    df_filt, _, _, de_dataset = test_fixture
    df_ph = df_filt[np.isin(df_filt["StopType"], [StopType.PH.value])]
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


def test_get_ctof(test_fixture):
    """Tests get_ctof function."""
    df_filt, _, _, _ = test_fixture
    df_filt = df_filt[df_filt["eTOF"].astype("str") != "FILL"]
    df_filt = df_filt[df_filt["cTOF"].astype("float") > 0]

    df_ph = df_filt[np.isin(df_filt["StopType"], [StopType.PH.value])]
    df_ssd = df_filt[np.isin(df_filt["StopType"], [StopType.SSD.value])]

    ph_ctof, ph_magnitude_v = get_ctof(
        df_ph["TOF"].astype("float").to_numpy(),
        df_ph["r"].astype("float").to_numpy(),
        "PH",
    )

    ssd_ctof, ssd_magnitude_v = get_ctof(
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
    np.testing.assert_allclose(
        ph_magnitude_v, df_ph["vmag"].astype("float"), atol=1e-01, rtol=0
    )
    np.testing.assert_allclose(
        ssd_magnitude_v, df_ssd["vmag"].astype("float"), atol=1e-01, rtol=0
    )


def test_determine_species(test_fixture):
    """Tests determine_species function."""
    df_filt, _, _, _ = test_fixture
    df_ph = df_filt[np.isin(df_filt["StopType"], [StopType.PH.value])]
    df_ssd = df_filt[np.isin(df_filt["StopType"], [StopType.SSD.value])]

    species_bin_ph = determine_species(
        df_ph["TOF"].astype("float").to_numpy(),
        df_ph["r"].astype("float").to_numpy(),
        "PH",
    )
    species_bin_ssd = determine_species(
        df_ssd["TOF"].astype("float").to_numpy(),
        df_ssd["r"].astype("float").to_numpy(),
        "SSD",
    )

    h_indices_ph = np.where(species_bin_ph == "H")[0]
    ctof_indices_ph = np.where(
        (df_ph["cTOF"].astype("float") > UltraConstants.CTOF_SPECIES_MIN)
        & (df_ph["cTOF"].astype("float") < UltraConstants.CTOF_SPECIES_MAX)
    )[0]

    h_indices_ssd = np.where(species_bin_ssd == "H")[0]
    ctof_indices_ssd = np.where(
        (df_ssd["cTOF"].astype("float") > UltraConstants.CTOF_SPECIES_MIN)
        & (df_ssd["cTOF"].astype("float") < UltraConstants.CTOF_SPECIES_MAX)
    )[0]

    np.testing.assert_array_equal(h_indices_ph, ctof_indices_ph)
    np.testing.assert_array_equal(h_indices_ssd, ctof_indices_ssd)


def test_get_phi_theta(test_fixture):
    """Tests get_phi_theta function."""
    df_filt, d, _, _ = test_fixture

    test_xf = df_filt["Xf"].astype("float").values
    test_yf = df_filt["Yf"].astype("float").values

    test_xb = df_filt["Xb"].astype("float").values
    test_yb = df_filt["Yb"].astype("float").values

    phi, theta = get_phi_theta((test_xf, test_yf), (test_xb, test_yb), d)
    expected_phi = df_filt["phi"].astype("float")
    expected_theta = df_filt["theta"].astype("float")

    np.testing.assert_allclose(phi, expected_phi, atol=1e-03, rtol=0)
    np.testing.assert_allclose(theta, expected_theta, atol=1e-03, rtol=0)


def test_get_eventtimes(test_fixture, use_fake_spin_data_for_time):
    """Tests get_eventtimes function."""
    df_filt, _, _, de_dataset = test_fixture
    # Create a spin table that cover spin 0-141
    use_fake_spin_data_for_time(0, 141 * 15)

    event_times, spin_starts, spin_period_sec = get_eventtimes(
        de_dataset["SPIN"].values, de_dataset["PHASE_ANGLE"].values
    )

    spin_df = get_spin_data()
    expected_min_df = spin_df[spin_df["spin_number"] == de_dataset["SPIN"].values.min()]
    expected_max_df = spin_df[spin_df["spin_number"] == de_dataset["SPIN"].values.max()]
    spin_period_sec_min = expected_min_df["spin_period_sec"].values[0]
    spin_period_sec_max = expected_max_df["spin_period_sec"].values[0]

    spin_start_min = (
        expected_min_df["spin_start_sec"] + expected_min_df["spin_start_subsec"] / 1000
    )
    spin_start_max = (
        expected_max_df["spin_start_sec"] + expected_max_df["spin_start_subsec"] / 1000
    )

    assert spin_start_min.values[0] == spin_starts.min()
    assert spin_start_max.values[0] == spin_starts.max()

    event_times_min = spin_start_min.values[0] + spin_period_sec_min * (
        de_dataset["PHASE_ANGLE"][0] / 720
    )
    event_times_max = spin_start_max.values[0] + spin_period_sec_max * (
        de_dataset["PHASE_ANGLE"][-1] / 720
    )

    assert event_times_min == event_times.min()
    assert event_times_max == event_times.max()
