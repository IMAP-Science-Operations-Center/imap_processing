"""Tests Spacecraft PSET for ULTRA L1c."""

import numpy as np
import pandas as pd
import pytest
import spiceypy
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.kernels import ensure_spice
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    get_de_energy_kev,
    get_de_velocity,
)
from imap_processing.ultra.l1c.spacecraft_pset import calculate_spacecraft_pset

TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.mark.external_test_data
@pytest.mark.external_kernel
@ensure_spice
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_pset():
    """Tests calculate_pset function."""
    # This is just setting up the data so that it is in the format of l1b_de_dataset.
    test_path = TEST_PATH / "ultra-90_raw_event_data_shortened.csv"
    df = pd.read_csv(test_path)
    instrument_velocity, _, _ = get_de_velocity(
        (df["Xf"], df["Yf"]), (df["Xb"], df["Yb"]), df["d"], df["TOF"].values
    )

    et = spiceypy.str2et(df["Epoch"].values)
    epoch = df["MET"].values

    frame_velocities = get_annotated_particle_velocity(
        et,
        instrument_velocity,
        SpiceFrame.IMAP_ULTRA_90,
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
    )

    particle_velocity_dps_spacecraft = frame_velocities[1]

    # Assume everything is Hydrogen
    species = np.full(len(particle_velocity_dps_spacecraft), "H", dtype="<U1")
    energy_dps_spacecraft = get_de_energy_kev(particle_velocity_dps_spacecraft, species)

    test_l1b_de_dataset = xr.Dataset(
        {
            "species": (["epoch"], species),
            "velocity_dps_sc": (
                ["epoch", "component"],
                particle_velocity_dps_spacecraft,
            ),
            "energy_spacecraft": (["epoch"], energy_dps_spacecraft),
        },
        coords={
            "epoch": ("epoch", epoch),
            "component": ("component", ["vx", "vy", "vz"]),
        },
    )

    spacecraft_pset = calculate_spacecraft_pset(
        test_l1b_de_dataset,
        test_l1b_de_dataset,  # placeholder for extendedspin_dataset
        test_l1b_de_dataset,  # placeholder for cullingmask_dataset
        "imap_ultra_l1c_45sensor-spacecraftpset",
    )
    assert "healpix" in spacecraft_pset.coords
    assert "epoch" in spacecraft_pset.coords
    assert "energy_bin_geometric_mean" in spacecraft_pset.coords
