"""Tests Spacecraft PSET for ULTRA L1c."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import spiceypy
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.time import met_to_sclkticks, sct_to_et
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    get_de_energy_kev,
    get_de_velocity,
    get_front_y_position,
)
from imap_processing.ultra.l1c.spacecraft_pset import calculate_spacecraft_pset
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.mark.external_test_data
@pytest.mark.external_kernel
def test_calculate_spacecraft_pset(
    random_spin_data,
    deadtime_datasets,
    imap_ena_sim_metakernel,
    use_fake_spin_data_for_time,
    ancillary_files,
    mock_spacecraft_pointing_lookups,
):
    """Tests calculate_spacecraft_pset function."""
    # Simulate a spin table from MET = 0 to MET = 141 * 15 seconds
    use_fake_spin_data_for_time(start_met=0, end_met=141 * 15)

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
            "e_bin": (["epoch"], np.ones(len(species), dtype=np.uint8)),
            "velocity_dps_sc": (
                ["epoch", "component"],
                particle_velocity_dps_spacecraft,
            ),
            "energy_spacecraft": (["epoch"], energy_dps_spacecraft),
            "spin_number": (["epoch"], df["Spin"].values),
            "quality_scattering": (
                ["epoch"],
                np.zeros(len(df["Spin"].values), dtype=np.uint16),
            ),
            "quality_outliers": (
                ["epoch"],
                np.zeros(len(df["Spin"].values), dtype=np.uint16),
            ),
            "event_times": sct_to_et(met_to_sclkticks(df["MET"].values)),
        },
        coords={
            "epoch": ("epoch", epoch),
            "component": ("component", ["vx", "vy", "vz"]),
        },
    )
    with (
        mock.patch(
            "imap_processing.ultra.l1c.spacecraft_pset.get_pointing_times",
            return_value=(482374890.0, 482374000.0),
        ),
        mock.patch(
            "imap_processing.ultra.l1c.ultra_l1c_pset_bins.ttj2000ns_to_met",
            side_effect=lambda x: x,
        ),
    ):
        spacecraft_pset = calculate_spacecraft_pset(
            test_l1b_de_dataset,
            test_l1b_de_dataset,  # placeholder for goodtimes_dataset
            deadtime_datasets["rates"],
            deadtime_datasets["params"],
            "imap_ultra_l1c_45sensor-spacecraftpset",
            ancillary_files,
            45,
            UltraConstants.TOFXPH_SPECIES_GROUPS["proton"],
        )
    assert "pixel_index" in spacecraft_pset.coords
    assert "epoch" in spacecraft_pset.coords
    assert "energy_bin_geometric_mean" in spacecraft_pset.coords


@pytest.mark.external_test_data
@pytest.mark.external_kernel
def test_calculate_spacecraft_pset_with_cdf(
    random_spin_data,
    ancillary_files,
    deadtime_datasets,
    imap_ena_sim_metakernel,
    use_fake_spin_data_for_time,
    mock_spacecraft_pointing_lookups,
):
    """Tests calculate_spacecraft_pset function with imported test data."""
    # Simulate a spin table from MET = 0 to MET = 141 * 15 seconds
    use_fake_spin_data_for_time(start_met=0, end_met=141 * 15)
    df = pd.read_csv(TEST_PATH / "IMAP-Ultra45_r1_L1_V0_shortened.csv")

    # Loop over all unique pointing numbers
    for pointing in df["pointing_number"].unique():
        df_subset = df[df["pointing_number"] == pointing].copy()

        de_dict = {}

        de_dict["epoch"] = df_subset["epoch"].values
        species_bin = np.full(len(df_subset), 1, dtype=np.uint8)

        # PosYSlit is True for left (start_type = 1)
        # PosYSlit is False for right (start_type = 2)
        start_type = np.where(df_subset["PosYSlit"].values, 1, 2)
        # Convert StartX, StopX, StopY to hundredths of mm.
        d, yf = get_front_y_position(
            start_type, df_subset["StopY"].values * 100, ancillary_files
        )
        tof_tenths_ns = df_subset["TOF"].values * 10000
        v, _, _ = get_de_velocity(
            (df_subset["StartX"].values * 100, yf),
            (df_subset["StopX"].values * 100, df_subset["StopY"].values * 100),
            d,
            tof_tenths_ns,
        )
        de_dict["direct_event_velocity"] = v.astype(np.float32)

        ultra_frame = SpiceFrame.IMAP_ULTRA_45
        _, sc_dps_velocity, _ = get_annotated_particle_velocity(
            df_subset["tdb"].values,
            de_dict["direct_event_velocity"],
            ultra_frame,
            SpiceFrame.IMAP_DPS,
            SpiceFrame.IMAP_SPACECRAFT,
        )

        de_dict["velocity_dps_sc"] = sc_dps_velocity
        de_dict["energy_spacecraft"] = get_de_energy_kev(sc_dps_velocity, species_bin)
        # Made up data for spin_number and energy_bin_geometric_mean
        de_dict["spin_number"] = np.full(len(sc_dps_velocity), 128)
        de_dict["energy_bin_geometric_mean"] = np.zeros(len(sc_dps_velocity))
        de_dict["quality_scattering"] = np.zeros(len(sc_dps_velocity), dtype=np.uint16)
        de_dict["quality_outliers"] = np.zeros(len(sc_dps_velocity), dtype=np.uint16)
        de_dict["e_bin"] = np.ones(len(sc_dps_velocity), dtype=np.uint8)
        de_dict["event_times"] = 817561854.185627 + (
            df_subset["tdb"].values - df_subset["tdb"].values[0]
        )

        name = "imap_ultra_l1b_45sensor-de"
        dataset = create_dataset(de_dict, name, "l1b")
        with (
            mock.patch(
                "imap_processing.ultra.l1c.spacecraft_pset.get_pointing_times",
                return_value=(482374890.0, 482374000.0),
            ),
            mock.patch(
                "imap_processing.ultra.l1c.ultra_l1c_pset_bins.ttj2000ns_to_met",
                side_effect=lambda x: x,
            ),
        ):
            spacecraft_pset = calculate_spacecraft_pset(
                dataset,
                dataset,  # placeholder for goodtimes_dataset
                deadtime_datasets["rates"],
                deadtime_datasets["params"],
                "imap_ultra_l1c_45sensor-spacecraftpset",
                ancillary_files,
                45,
                UltraConstants.TOFXPH_SPECIES_GROUPS["proton"],
            )
        # TODO: validate with output histogram data once we have it in healpix.
        assert (
            spacecraft_pset.attrs["Logical_source"]
            == "imap_ultra_l1c_45sensor-spacecraftpset"
        )
