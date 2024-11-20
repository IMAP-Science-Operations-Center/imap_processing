"""Tests Extended Raw Events for ULTRA L1b."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.l1b.de import calculate_de


@pytest.fixture()
def df_filt(de_dataset, events_fsw_comparison_theta_0):
    """Fixture to import test dataset."""
    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    df_filt = df_filt.replace("FILL", 0)

    return df_filt


@mock.patch("imap_processing.ultra.l1b.de.get_annotated_particle_velocity")
def test_calculate_de(mock_get_annotated_particle_velocity, de_dataset, df_filt):
    """Tests calculate_de function."""

    # Mock get_annotated_particle_velocity to avoid needing kernels
    def side_effect_func(event_times, position, ultra_frame, dps_frame, sc_frame):
        """
        Mock behavior of get_annotated_particle_velocity.

        Returns NaN-filled arrays matching the expected output shape.
        """
        num_events = event_times.size
        return (
            np.full((num_events, 3), np.nan),  # sc_velocity
            np.full((num_events, 3), np.nan),  # sc_dps_velocity
            np.full((num_events, 3), np.nan),  # helio_velocity
        )

    mock_get_annotated_particle_velocity.side_effect = side_effect_func

    dataset = calculate_de(de_dataset, "imap_ultra_l1b_45sensor-de")

    # Front and back positions
    assert np.allclose(dataset["x_front"].data, df_filt["Xf"].astype("float"))
    assert np.allclose(dataset["y_front"], df_filt["Yf"].astype("float"))
    assert np.allclose(dataset["x_back"], df_filt["Xb"].astype("float"))
    assert np.allclose(dataset["y_back"], df_filt["Yb"].astype("float"))

    # Coincidence positions
    assert np.allclose(dataset["x_coin"], df_filt["Xc"].astype("float"))

    # Time of flight
    assert np.allclose(dataset["tof_start_stop"], df_filt["TOF"].astype("float"))
    assert np.allclose(dataset["tof_stop_coin"], df_filt["eTOF"].astype("float"))
    assert np.allclose(dataset["tof_corrected"], df_filt["cTOF"].astype("float"))

    # Distances and path lengths
    assert np.allclose(dataset["front_back_distance"], df_filt["d"].astype("float"))
    assert np.allclose(dataset["path_length"], df_filt["r"].astype("float"))

    # Coincidence, start, and event types
    assert np.allclose(dataset["coincidence_type"], df_filt["CoinType"].astype("float"))
    assert np.allclose(dataset["start_type"], df_filt["StartType"].astype("float"))
    assert np.allclose(dataset["event_type"], df_filt["StopType"].astype("float"))

    # Energies and species
    assert np.allclose(dataset["energy"], df_filt["Energy"].astype("float"))
    assert np.allclose(
        dataset["species"], np.full(len(de_dataset["epoch"]), np.nan, dtype=np.uint8)
    )

    # Velocities in various frames
    test_tof = dataset["tof_start_stop"]
    assert np.allclose(
        dataset["vx_ultra"][test_tof > 0],
        -df_filt["vhatX"].astype("float").values[test_tof > 0],
        rtol=1e-2,
    )
    assert np.allclose(
        dataset["vy_ultra"][test_tof > 0],
        -df_filt["vhatY"].astype("float").values[test_tof > 0],
        rtol=1e-2,
    )
    assert np.allclose(
        dataset["vz_ultra"][test_tof > 0],
        -df_filt["vhatZ"].astype("float").values[test_tof > 0],
        rtol=1e-2,
    )

    assert dataset["vx_sc"].shape == (len(de_dataset["epoch"]),)
    assert dataset["vy_sc"].shape == (len(de_dataset["epoch"]),)
    assert dataset["vz_sc"].shape == (len(de_dataset["epoch"]),)

    assert dataset["vx_dps_sc"].shape == (len(de_dataset["epoch"]),)
    assert dataset["vy_dps_sc"].shape == (len(de_dataset["epoch"]),)
    assert dataset["vz_dps_sc"].shape == (len(de_dataset["epoch"]),)

    assert dataset["vx_dps_helio"].shape == (len(de_dataset["epoch"]),)
    assert dataset["vy_dps_helio"].shape == (len(de_dataset["epoch"]),)
    assert dataset["vz_dps_helio"].shape == (len(de_dataset["epoch"]),)

    # Event efficiency
    assert np.allclose(
        dataset["event_efficiency"],
        np.full(len(de_dataset["epoch"]), np.nan),
        equal_nan=True,
    )
