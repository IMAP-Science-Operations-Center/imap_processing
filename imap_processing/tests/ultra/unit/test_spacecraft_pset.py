"""Tests Spacecraft PSET for ULTRA L1c."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import spiceypy
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.time import met_to_sclkticks, sct_to_et
from imap_processing.tests.conftest import _download_external_data
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
    aux_dataset,
    rates_dataset,
    imap_ena_sim_metakernel,
    use_fake_spin_data_for_time,
    ancillary_files,
    mock_spacecraft_pointing_lookups,
):
    """Tests calculate_spacecraft_pset function."""
    # Simulate a spin table from MET = 0 to MET = 141 * 15 seconds
    use_fake_spin_data_for_time(start_met=0, end_met=141 * 15)
    # Ensure rate and aux data have the correct time range
    t_rates = np.linspace(0, 141 * 15, len(rates_dataset.shcoarse.data))
    rates_dataset.shcoarse.data = t_rates
    aux_dataset.timespinstart.data = t_rates[: len(aux_dataset.timespinstart.data)]
    aux_dataset.timespinstart.data[-1] = t_rates[-1]
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
            "ebin": (["epoch"], np.ones(len(species), dtype=np.uint8)),
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
        attrs={"Repointing": "repoint00001"},
    )
    with mock.patch(
        "imap_processing.ultra.l1c.spacecraft_pset.get_pointing_times_from_id",
        return_value=(482374890.0, 482374000.0),
    ):
        spacecraft_pset = calculate_spacecraft_pset(
            test_l1b_de_dataset,
            test_l1b_de_dataset,  # placeholder for goodtimes_dataset
            rates_dataset,
            aux_dataset,
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
    ancillary_files,
    aux_dataset,
    rates_dataset,
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
        # Ensure rate and aux data have the correct time range
        t_rates = np.linspace(0, 141 * 15, len(rates_dataset.shcoarse.data))
        rates_dataset.shcoarse.data = t_rates
        aux_dataset.timespinstart.data = t_rates[: len(aux_dataset.timespinstart.data)]
        aux_dataset.timespinstart.data[-1] = t_rates[-1]
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
        de_dict["ebin"] = np.ones(len(sc_dps_velocity), dtype=np.uint8)
        de_dict["event_times"] = 817561854.185627 + (
            df_subset["tdb"].values - df_subset["tdb"].values[0]
        )

        name = "imap_ultra_l1b_45sensor-de"
        dataset = create_dataset(de_dict, name, "l1b")
        dataset.attrs["Repointing"] = "repoint00000"
        with mock.patch(
            "imap_processing.ultra.l1c.spacecraft_pset.get_pointing_times_from_id",
            return_value=(472374890.0, 582378000.0),
        ):
            spacecraft_pset = calculate_spacecraft_pset(
                dataset,
                dataset,  # placeholder for goodtimes_dataset
                rates_dataset,
                aux_dataset,
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


@pytest.mark.skip(reason="Long running test for validation purposes.")
def test_validate_exposure_time_and_sensitivities(
    ancillary_files, rates_dataset, aux_dataset
):
    """Validates exposure time and sensitivities for ebin 0."""
    test_data = [
        (
            "imap_ultra_l1c-90sensor-sc-pointing-theta_20250101_v001.csv",
            "ultra/data/l1/",
        ),
        ("imap_ultra_l1c-90sensor-sc-pointing-phi_20250101_v001.csv", "ultra/data/l1/"),
        (
            "imap_ultra_l1c-90sensor-sc-pointing-index_20250101_v001.csv",
            "ultra/data/l1/",
        ),
        ("imap_ultra_l1c-90sensor-sc-pointing-bsf_20250101_v001.csv", "ultra/data/l1/"),
        ("Exposures-IMAP_ULTRA_90-IMAP_DPS-SC-nside32-ebin0.csv", "ultra/data/l1/"),
        ("SENS-IMAP_ULTRA_90-IMAP_DPS-SC-nside32-ebin0.csv", "ultra/data/l1/"),
        ("imap_ultra_l1b_45sensor-de_20000101-repoint00000_v000.cdf", "ultra/data/l1/"),
    ]
    _download_external_data(test_data)
    l1b_de = TEST_PATH / "imap_ultra_l1b_45sensor-de_20000101-repoint00000_v000.cdf"
    l1b_de = load_cdf(l1b_de)
    sensitivities_ebin_0 = pd.read_csv(
        TEST_PATH / "SENS-IMAP_ULTRA_90-IMAP_DPS-SC-nside32-ebin0.csv"
    )
    exposure_factor_ebin_0 = pd.read_csv(
        TEST_PATH / "Exposures-IMAP_ULTRA_90-IMAP_DPS-SC-nside32-ebin0.csv"
    )
    test_deadtimes = (
        pd.read_csv(TEST_PATH / "test_p0_ebin0_deadtimes.csv", header=None)
        .to_numpy()
        .squeeze()
    )
    npix = 12288  # nside 32
    # Create a minimal dataset to pass to the function
    dataset = xr.Dataset(
        {
            "spin_number": (["epoch"], np.array([1, 2, 3])),
        }
    )
    dataset.attrs["Repointing"] = "repoint00000"

    ancillary_files["l1c-90sensor-sc-pointing-theta"] = (
        TEST_PATH / "imap_ultra_l1c-90sensor-sc-pointing-theta_20250101_v001.csv"
    )
    ancillary_files["l1c-90sensor-sc-pointing-phi"] = (
        TEST_PATH / "imap_ultra_l1c-90sensor-sc-pointing-phi_20250101_v001.csv"
    )
    ancillary_files["l1c-90sensor-sc-pointing-index"] = (
        TEST_PATH / "imap_ultra_l1c-90sensor-sc-pointing-index_20250101_v001.csv"
    )
    ancillary_files["l1c-90sensor-sc-pointing-bsf"] = (
        TEST_PATH / "imap_ultra_l1c-90sensor-sc-pointing-bsf_20250101_v001.csv"
    )

    pointing_range_met = (472374890.0, 582378000.0)
    # Create mock spin data that has 5525 nominal spins
    # Create DataFrame
    nspins = 5522
    nominal_spin_seconds = 15.0
    spin_data = pd.DataFrame(
        {
            "spin_start_met": np.linspace(
                pointing_range_met[0], pointing_range_met[1], nspins
            ),
            "spin_period_sec": np.full(nspins, nominal_spin_seconds),
            "spin_phase_valid": np.ones(nspins),
            "spin_period_valid": np.ones(nspins),
        }
    )
    with (
        # Mock the pointing times
        mock.patch(
            "imap_processing.ultra.l1c.spacecraft_pset.get_pointing_times_from_id",
            return_value=pointing_range_met,
        ),
        # Mock deadtimes to be all ones
        mock.patch(
            "imap_processing.ultra.l1c.ultra_l1c_pset_bins."
            "get_deadtime_ratios_by_spin_phase",
            return_value=xr.DataArray(test_deadtimes, dims="spin_phase_step"),
        ),
        # Mock spin data to match nominal spins in a pointing period
        mock.patch(
            "imap_processing.ultra.l1c.ultra_l1c_pset_bins.get_spin_data",
            return_value=spin_data,
        ),
        # Mock background rates to be constant 0.1
        mock.patch(
            "imap_processing.ultra.l1c.spacecraft_pset.get_spacecraft_background_rates",
            return_value=np.ones((46, npix)),
        ),
        # Mock culling mask (no culling)
        mock.patch("imap_processing.ultra.l1c.spacecraft_pset.compute_culling_mask"),
    ):
        pset = calculate_spacecraft_pset(
            l1b_de,
            dataset,
            rates_dataset,
            aux_dataset,
            "imap_ultra_l1c_90sensor-spacecraftpset",
            ancillary_files,
            90,
            UltraConstants.TOFXPH_SPECIES_GROUPS["proton"],
        )

    # Validate exposure times for ebin 0
    exposure_times = pset["exposure_factor"][0, 0, :].values
    expected_exposure_times = exposure_factor_ebin_0["P0"].to_numpy()
    np.testing.assert_allclose(
        exposure_times,
        expected_exposure_times,
        rtol=1e-8,
        err_msg="Exposure times do not match expected values for ebin 0.",
    )
    # Validate sensitivities for ebin 0
    sensitivity = pset["sensitivity"][0, :].values
    expected_sensitivity = sensitivities_ebin_0["Sensitivity (cm2)"].to_numpy()
    np.testing.assert_allclose(
        sensitivity,
        expected_sensitivity,
        rtol=0.15,
        err_msg="Sensitivities times do not match expected values for ebin 0.",
    )
