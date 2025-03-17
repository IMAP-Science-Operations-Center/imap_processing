"""Tests Extended Raw Events for ULTRA L1b."""

import numpy as np
import pandas as pd
import pytest

from imap_processing.spice.geometry import SpiceFrame
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import get_de_velocity, get_de_energy_kev


@pytest.fixture()
def df_filt(de_dataset, events_fsw_comparison_theta_0):
    """Fixture to import test dataset."""
    df = pd.read_csv(events_fsw_comparison_theta_0)
    df_filt = df[df["StartType"] != -1]
    df_filt = df_filt.replace("FILL", 0)

    return df_filt


@pytest.mark.external_test_data
def test_pset(_download_test_data):
    """Tests calculate_de function."""

    print("hi")

    instrument_velocity = get_de_velocity(front_position,
                                          back_position, d,
                                          tof)
    frame_velocities = get_annotated_particle_velocity(time,
                                                       instrument_velocity,
                                                       SpiceFrame.IMAP_ULTRA_45,
                                                       SpiceFrame.IMAP_DPS,
                                                       SpiceFrame.IMAP_SPACECRAFT,)

    # Assume everything is Hydrogen
    species = np.full(len(frame_velocities[1]), "H", dtype="<U1")
    energy = get_de_energy_kev(frame_velocities[1], species)

    # Create fake de dataset


