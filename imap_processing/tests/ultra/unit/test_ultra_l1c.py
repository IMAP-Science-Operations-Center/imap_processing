from unittest import mock

import astropy_healpix.healpy as hp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.ultra.l1b.ultra_l1b_annotated import (
    get_annotated_particle_velocity,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    get_de_energy_kev,
    get_de_velocity,
    get_front_y_position,
)
from imap_processing.ultra.l1c.ultra_l1c import ultra_l1c
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.fixture
def mock_spacecraft_pointing_lookups():
    """Test lookup tables fixture."""
    pix = hp.nside2npix(128)
    steps = 10  # Reduced for testing
    for_indices_by_spin_phase = np.random.choice(
        [True, False], size=(pix, steps), p=[0.1, 0.9]
    )
    theta_vals = np.random.uniform(-60, 60, size=(pix, steps))
    phi_vals = np.random.uniform(-60, 60, size=(pix, steps))
    ra_and_dec = np.random.uniform(-80, 80, size=(pix, 2))
    boundary_scale_factors = np.ones((pix, steps))
    with (
        mock.patch(
            "imap_processing.ultra.l1c.spacecraft_pset.get_spacecraft_pointing_lookup_tables"
        ) as mock_lookup,
        mock.patch(
            "imap_processing.ultra.l1c.helio_pset.get_spacecraft_pointing_lookup_tables"
        ) as mock_lookup_helio,
    ):
        mock_lookup.return_value = (
            for_indices_by_spin_phase,
            theta_vals,
            phi_vals,
            ra_and_dec,
            boundary_scale_factors,
        )
        mock_lookup_helio.return_value = (
            for_indices_by_spin_phase,
            theta_vals,
            phi_vals,
            ra_and_dec,
            boundary_scale_factors,
        )
        yield mock_lookup


@pytest.fixture
def fake_spin_data(spice_test_data_path, use_test_spin_data_csv):
    """Generate fake spin dataframe for testing"""
    fake_spin_path = spice_test_data_path / "fake_spin_data.csv"
    use_test_spin_data_csv([fake_spin_path])
    return fake_spin_path


@pytest.fixture
def mock_data_l1b_dict():
    # Create sample data for the xarray Dataset
    epoch = np.arange(
        "2024-02-07T15:28:37", "2024-02-07T15:28:42", dtype="datetime64[s]"
    ).astype("datetime64[ns]")

    data_vars_histogram = {
        "sid": ("epoch", np.zeros(5)),
        "row": ("epoch", np.zeros(5)),
        "column": ("epoch", np.zeros(5)),
        "shcoarse": ("epoch", np.zeros(5)),
        "spin": ("epoch", np.zeros(5)),
        "packetdata": ("epoch", np.zeros(5)),
    }

    coords = {"epoch": epoch}

    attrs_histogram = {
        "Logical_source": "imap_ultra_l1a_45sensor-histogram",
        "Logical_source_description": "IMAP Mission ULTRA Instrument "
        "Level-1A Single-Sensor Data",
    }

    dataset_histogram = xr.Dataset(
        data_vars=data_vars_histogram, coords=coords, attrs=attrs_histogram
    )

    data_vars_cullingmask = {
        "spin_number": ("epoch", np.zeros(5)),
    }

    attrs_cullingmask = {
        "Logical_source": "imap_ultra_l1b_45sensor-cullingmask",
        "Logical_source_description": "IMAP Mission ULTRA Instrument "
        "Level-1B Culling Mask Data",
    }

    dataset_cullingmask = xr.Dataset(
        data_vars_cullingmask, coords={"epoch": epoch}, attrs=attrs_cullingmask
    )

    data_dict = {
        "imap_ultra_l1b_45sensor-cullingmask": dataset_cullingmask,
        "imap_ultra_l1a_45sensor-histogram": dataset_histogram,
    }
    return data_dict


@pytest.fixture
def mock_data_l1c_dict():
    epoch = np.array(
        [760591786368000000, 760591787368000000, 760591788368000000],
        dtype="datetime64[ns]",
    )
    data_dict = {"epoch": epoch, "sid": np.zeros(3)}
    return data_dict


def test_create_dataset(mock_data_l1c_dict):
    """Tests that dataset is created as expected."""
    dataset = create_dataset(
        mock_data_l1c_dict, "imap_ultra_l1c_45sensor-histogram", "l1c"
    )

    assert "epoch" in dataset.coords
    assert dataset.coords["epoch"].dtype == "datetime64[ns]"
    assert dataset.attrs["Logical_source"] == "imap_ultra_l1c_45sensor-histogram"
    assert dataset["sid"].attrs["UNITS"] == " "
    np.testing.assert_array_equal(dataset["sid"], np.zeros(3))


def test_ultra_l1c(mock_data_l1b_dict):
    """Tests that L1c data is created."""
    path = imap_module_directory / "tests" / "ultra" / "data" / "l1"
    ancillary_files = {
        "l1c-90sensor-dps-exposure": path
        / "imap_ultra_l1c-90sensor-dps-exposure_20250101_v000.csv",
        "l1c-90sensor-efficiencies": path
        / "imap_ultra_l1c-90sensor-efficiencies_20250101_v000.csv",
        "l1c-90sensor-gf": path / "imap_ultra_l1c-90sensor-gf_20250101_v000.csv",
    }

    output_datasets = ultra_l1c(mock_data_l1b_dict, ancillary_files, has_spice=False)

    assert len(output_datasets) == 1
    assert (
        output_datasets[0].attrs["Logical_source"]
        == "imap_ultra_l1c_45sensor-histogram"
    )
    assert (
        output_datasets[0].attrs["Logical_source_description"]
        == "IMAP-Ultra Instrument Level-1C Pointing Set Grid Histogram Data."
    )


def test_ultra_l1c_error(mock_data_l1b_dict):
    """Tests that L1b data throws an error."""
    mock_data_l1b_dict["bad_key"] = mock_data_l1b_dict.pop(
        "imap_ultra_l1a_45sensor-histogram"
    )
    ancillary_files = {}
    with pytest.raises(
        ValueError, match="Data dictionary does not contain the expected keys."
    ):
        ultra_l1c(mock_data_l1b_dict, ancillary_files, has_spice=False)


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
    """Tests ultra_l1c function with imported test data."""
    # Simulate a spin table from MET = 0 to MET = 141 * 15 seconds
    use_fake_spin_data_for_time(start_met=0, end_met=141 * 15)
    df = pd.read_csv(TEST_PATH / "IMAP-Ultra45_r1_L1_V0_shortened.csv")

    # Select a single pointing number
    pointing = 0
    df_subset = df[df["pointing_number"] == pointing].copy()

    de_dict = {}

    de_dict["epoch"] = df_subset["epoch"].values
    species_bin = np.full(len(df_subset), 1, dtype=np.uint8)

    # PosYSlit is True for left (start_type = 1)
    # PosYSlit is False for right (start_type = 2)
    start_type = np.where(df_subset["PosYSlit"].values, 1, 2)
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
    de_dict["quality_scattering"] = np.zeros(len(v), dtype=np.uint16)
    de_dict["quality_outliers"] = np.zeros(len(v), dtype=np.uint16)

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

    name = "imap_ultra_l1b_45sensor-de"
    dataset = create_dataset(de_dict, name, "l1b")

    data_dict = {
        "imap_ultra_l1b_45sensor-de": dataset,
        "imap_ultra_l1b_45sensor-extendedspin": dataset,  # placeholder
        "imap_ultra_l1b_45sensor-cullingmask": dataset,  # placeholder
        "imap_ultra_l1a_45sensor-rates": deadtime_datasets["rates"],
        "imap_ultra_l1a_45sensor-params": deadtime_datasets["params"],
    }

    output_datasets = ultra_l1c(data_dict, ancillary_files, has_spice=False)
    output_datasets[0].attrs["Data_version"] = "999"
    output_datasets[0].attrs["Repointing"] = f"repoint{pointing + 1:05d}"
    test_data_path = write_cdf(output_datasets[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1c_45sensor-spacecraftpset_20250415-repoint00001_v999.cdf"
    )


@pytest.mark.external_test_data
@pytest.mark.external_kernel
def test_calculate_helio_pset_with_cdf(
    random_spin_data,
    ancillary_files,
    imap_ena_sim_metakernel,
    mock_spacecraft_pointing_lookups,
    deadtime_datasets,
    use_fake_spin_data_for_time,
    use_fake_repoint_data_for_time,
):
    """Tests ultra_l1c function with imported test data."""

    # Simulate a spin table from MET = 0 to MET = 141 * 15 seconds
    use_fake_spin_data_for_time(start_met=0, end_met=141 * 15)
    use_fake_repoint_data_for_time(np.arange(4.32374e08, 4.99374e08 + 10, 10))
    df = pd.read_csv(TEST_PATH / "IMAP-Ultra45_r1_L1_V0_shortened.csv")

    # Select a single pointing number
    pointing = 0
    df_subset = df[df["pointing_number"] == pointing].copy()

    de_dict = {}

    de_dict["epoch"] = df_subset["epoch"].values
    # Fake SCLK in seconds that matches SPICE.
    de_dict["event_times"] = np.full(len(df_subset), 2.41187e13)
    species_bin = np.full(len(df_subset), 1, dtype=np.uint8)

    # PosYSlit is True for left (start_type = 1)
    # PosYSlit is False for right (start_type = 2)
    start_type = np.where(df_subset["PosYSlit"].values, 1, 2)
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
    _, _, helio_dps_velocity = get_annotated_particle_velocity(
        df_subset["tdb"].values,
        de_dict["direct_event_velocity"],
        ultra_frame,
        SpiceFrame.IMAP_DPS,
        SpiceFrame.IMAP_SPACECRAFT,
    )

    de_dict["velocity_dps_helio"] = helio_dps_velocity
    de_dict["energy_heliosphere"] = get_de_energy_kev(helio_dps_velocity, species_bin)
    de_dict["quality_scattering"] = np.zeros(len(helio_dps_velocity), dtype=np.uint16)
    de_dict["quality_outliers"] = np.zeros(len(helio_dps_velocity), dtype=np.uint16)

    name = "imap_ultra_l1b_45sensor-de"
    dataset = create_dataset(de_dict, name, "l1b")

    data_dict = {
        "imap_ultra_l1b_45sensor-de": dataset,
        "imap_ultra_l1b_45sensor-extendedspin": xr.Dataset(),  # placeholder
        "imap_ultra_l1b_45sensor-cullingmask": xr.Dataset(),  # placeholder
        "imap_ultra_l1a_45sensor-rates": deadtime_datasets["rates"],
        "imap_ultra_l1a_45sensor-params": deadtime_datasets["params"],
    }

    output_datasets = ultra_l1c(data_dict, ancillary_files, has_spice=True)
    output_datasets[0].attrs["Data_version"] = "999"
    output_datasets[0].attrs["Repointing"] = f"repoint{pointing + 1:05d}"
    test_data_path = write_cdf(output_datasets[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1c_45sensor-heliopset_20250415-repoint00001_v999.cdf"
    )
