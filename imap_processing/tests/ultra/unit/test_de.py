"""Tests Extended Raw Events for ULTRA L1b."""

import numpy as np
import pandas as pd
import pytest

from imap_processing.ultra.constants import UltraConstants


@pytest.fixture()
def df_filt(de_dataset, events_fsw_comparison_theta_0):
    """Fixture to import test dataset."""
    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    df_filt = df_filt.replace("FILL", 0)

    return df_filt


def test_calculate_de(l1b_de_dataset, df_filt):
    """Tests calculate_de function."""

    l1b_de_dataset = l1b_de_dataset[0]
    l1b_de_dataset = l1b_de_dataset.where(
        l1b_de_dataset["start_type"] != np.iinfo(np.int64).min, drop=True
    )
    # Front and back positions
    assert np.allclose(l1b_de_dataset["x_front"].data, df_filt["Xf"].astype("float"))
    assert np.allclose(l1b_de_dataset["y_front"], df_filt["Yf"].astype("float"))
    assert np.allclose(l1b_de_dataset["x_back"], df_filt["Xb"].astype("float"))
    assert np.allclose(l1b_de_dataset["y_back"], df_filt["Yb"].astype("float"))

    # Coincidence positions
    assert np.allclose(l1b_de_dataset["x_coin"], df_filt["Xc"].astype("float"))

    # Time of flight
    assert np.allclose(l1b_de_dataset["tof_start_stop"], df_filt["TOF"].astype("float"))
    assert np.allclose(l1b_de_dataset["tof_stop_coin"], df_filt["eTOF"].astype("float"))
    assert np.allclose(l1b_de_dataset["tof_corrected"], df_filt["cTOF"].astype("float"))

    # Distances and path lengths
    assert np.allclose(
        l1b_de_dataset["front_back_distance"], df_filt["d"].astype("float")
    )
    assert np.allclose(l1b_de_dataset["path_length"], df_filt["r"].astype("float"))

    # Coincidence, start, and event types
    assert np.allclose(
        l1b_de_dataset["coincidence_type"], df_filt["CoinType"].astype("float")
    )
    assert np.allclose(
        l1b_de_dataset["start_type"], df_filt["StartType"].astype("float")
    )
    assert np.allclose(
        l1b_de_dataset["event_type"], df_filt["StopType"].astype("float")
    )

    # Energies and species
    assert np.allclose(l1b_de_dataset["energy"], df_filt["Energy"].astype("float"))
    species_array = l1b_de_dataset["species"][
        np.where(
            (l1b_de_dataset["tof_corrected"] > UltraConstants.CTOF_SPECIES_MIN)
            & (l1b_de_dataset["tof_corrected"] < UltraConstants.CTOF_SPECIES_MAX)
        )[0]
    ]
    assert np.all(species_array == "H")

    # Velocities in various frames
    test_tof = l1b_de_dataset["tof_start_stop"]
    test_ph = l1b_de_dataset["event_type"]
    test_species = l1b_de_dataset["species"]
    condition = (test_tof > 0) & (test_ph < 8)
    assert np.allclose(
        l1b_de_dataset["direct_event_velocity"][:, 0].values[condition],
        df_filt["vx"].astype("float").values[condition],
        rtol=1e-2,
    )
    assert np.allclose(
        l1b_de_dataset["direct_event_velocity"][:, 1].values[condition],
        df_filt["vy"].astype("float").values[condition],
        rtol=1e-2,
    )
    assert np.allclose(
        l1b_de_dataset["direct_event_velocity"][:, 2].values[condition],
        df_filt["vz"].astype("float").values[condition],
        rtol=1e-2,
    )
    assert np.allclose(
        l1b_de_dataset["velocity_magnitude"].values[condition],
        df_filt["vmag"].astype("float").values[condition],
        rtol=1e-2,
    )
    condition = (test_tof > 0) & (test_ph < 8) & (test_species == "H")
    assert np.allclose(
        l1b_de_dataset["tof_energy"].values[condition],
        df_filt["energy_revised"].astype("float").values[condition],
        rtol=1e-2,
    )
    assert np.allclose(
        l1b_de_dataset["phi"].values,
        df_filt["phi"].astype("float").values,
        rtol=1e-2,
    )
    assert np.allclose(
        l1b_de_dataset["theta"].values,
        df_filt["theta"].astype("float").values,
        rtol=1e-2,
    )

    assert l1b_de_dataset["velocity_sc"].shape == (len(l1b_de_dataset["epoch"]), 3)
    assert l1b_de_dataset["velocity_dps_sc"].shape == (len(l1b_de_dataset["epoch"]), 3)
    assert l1b_de_dataset["velocity_dps_helio"].shape == (
        len(l1b_de_dataset["epoch"]),
        3,
    )

    # Event efficiency
    assert np.allclose(
        l1b_de_dataset["event_efficiency"],
        np.full(len(l1b_de_dataset["epoch"]), np.nan),
        equal_nan=True,
    )
