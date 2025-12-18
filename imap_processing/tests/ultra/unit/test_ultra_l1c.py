from unittest import mock

import astropy_healpix.healpy as hp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.time import (
    et_to_met,
)
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

    data_vars_goodtimes = {
        "spin_number": ("epoch", np.zeros(5)),
    }

    attrs_goodtimes = {
        "Logical_source": "imap_ultra_l1b_45sensor-goodtimes",
        "Logical_source_description": "IMAP Mission ULTRA Instrument "
        "Level-1B Culling Mask Data",
    }

    dataset_goodtimes = xr.Dataset(
        data_vars_goodtimes, coords={"epoch": epoch}, attrs=attrs_goodtimes
    )

    data_dict = {
        "imap_ultra_l1b_45sensor-goodtimes": dataset_goodtimes,
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


def test_ultra_l1c_error(mock_data_l1b_dict):
    """Tests that L1b data throws an error."""
    mock_data_l1b_dict["bad_key"] = mock_data_l1b_dict.pop(
        "imap_ultra_l1a_45sensor-histogram"
    )
    ancillary_files = {}
    with pytest.raises(
        ValueError, match="Data dictionary does not contain the expected keys."
    ):
        ultra_l1c(mock_data_l1b_dict, ancillary_files, "45sensor-spacecraftpset")


@pytest.mark.external_test_data
@pytest.mark.external_kernel
def test_calculate_spacecraft_pset_with_cdf(
    ancillary_files,
    rates_dataset,
    aux_dataset,
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
    # Ensure rate and aux data are in the correct time window
    t_rates = np.linspace(0, 141 * 15, len(rates_dataset.shcoarse.data))
    rates_dataset.shcoarse.data = t_rates
    aux_dataset.timespinstart.data = t_rates[: len(aux_dataset.timespinstart.data)]
    aux_dataset.timespinstart.data[-1] = t_rates[-1]
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
    de_dict["species"] = np.ones(len(sc_dps_velocity), dtype=np.uint8)
    de_dict["ebin"] = np.ones(len(sc_dps_velocity), dtype=np.uint8)
    de_dict["event_times"] = df_subset["tdb"].values

    name = "imap_ultra_l1b_45sensor-de"
    dataset = create_dataset(de_dict, name, "l1b")
    dataset.attrs["Repointing"] = "repoint00001"
    data_dict = {
        "imap_ultra_l1b_45sensor-de": dataset,
        "imap_ultra_l1b_45sensor-extendedspin": dataset,  # placeholder
        "imap_ultra_l1b_45sensor-goodtimes": dataset,  # placeholder
        "imap_ultra_l1a_45sensor-rates": rates_dataset,
        "imap_ultra_l1a_45sensor-aux": aux_dataset,
    }
    with (
        mock.patch(
            "imap_processing.ultra.l1c.spacecraft_pset.get_pointing_times_from_id",
            return_value=(482374890.0, 482374000.0),
        ),
    ):
        output_datasets = ultra_l1c(
            data_dict, ancillary_files, "45sensor-spacecraftpset"
        )
    output_datasets[0].attrs["Data_version"] = "999"
    output_datasets[0].attrs["Repointing"] = f"repoint{pointing + 1:05d}"
    output_datasets[0].attrs["Start_date"] = "20250415"
    test_data_path = write_cdf(output_datasets[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1c_45sensor-spacecraftpset_20250415-repoint00001_v999.cdf"
    )


@pytest.mark.external_test_data
@pytest.mark.external_kernel
def test_calculate_helio_pset_with_cdf(
    ancillary_files,
    imap_ena_sim_metakernel,
    mock_helio_pointing_lookups,
    aux_dataset,
    rates_dataset,
    use_fake_spin_data_for_time,
):
    """Tests ultra_l1c function with imported test data."""
    # Simulate a spin table from MET = 0 to MET = 141 * 15 seconds
    use_fake_spin_data_for_time(
        start_met=et_to_met(817561854.185627),
        end_met=et_to_met(817561854.185627 + 141 * 15),
    )
    df = pd.read_csv(TEST_PATH / "IMAP-Ultra45_r1_L1_V0_shortened.csv")

    # Ensure rate and aux data are in the correct time window
    t_off = 56970125
    rates_dataset.shcoarse.data += t_off
    aux_dataset.timespinstart.data += t_off
    # Select a single pointing number
    pointing = 0
    df_subset = df[df["pointing_number"] == pointing].copy()
    de_dict = {}

    de_dict["epoch"] = df_subset["epoch"].values
    # Fake SCLK in seconds that matches SPICE.
    de_dict["event_times"] = np.full(len(df_subset), 2.41187e13)
    de_dict["ebin"] = np.ones(len(df_subset), dtype=np.uint8)
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
    de_dict["species"] = np.ones(len(helio_dps_velocity), dtype=np.uint8)
    de_dict["event_times"] = df_subset["tdb"].values
    name = "imap_ultra_l1b_45sensor-de"
    dataset = create_dataset(de_dict, name, "l1b")
    dataset.attrs["Repointing"] = "repoint00001"
    data_dict = {
        "imap_ultra_l1b_45sensor-de": dataset,
        "imap_ultra_l1b_45sensor-extendedspin": xr.Dataset(),  # placeholder
        "imap_ultra_l1b_45sensor-goodtimes": xr.Dataset(
            {
                "spin_number": ("epoch", np.zeros(5)),
            }
        ),  # placeholder
        "imap_ultra_l1a_45sensor-rates": rates_dataset,
        "imap_ultra_l1a_45sensor-aux": aux_dataset,
    }
    n_pix = hp.nside2npix(32)
    mock_eff = np.ones((46, n_pix))
    mock_gf = mock_eff * 2
    with (
        mock.patch(
            "imap_processing.ultra.l1c.helio_pset.get_pointing_times_from_id",
            return_value=(482374890.0, 482374000.0),
        ),
        # Mock efficiencies and geometric function to known values
        mock.patch(
            "imap_processing.ultra.l1c.helio_pset.get_efficiencies_and_geometric_function",
            return_value=(mock_gf, mock_eff),
        ),
    ):
        output_datasets = ultra_l1c(data_dict, ancillary_files, "45sensor-heliopset")
    output_datasets[0].attrs["Data_version"] = "999"
    output_datasets[0].attrs["Repointing"] = f"repoint{pointing + 1:05d}"

    # Although the arrays are different, their sums should be equal. The helio adjusted
    # ones are just rebinned.
    np.testing.assert_array_equal(
        np.sum(output_datasets[0]["efficiency"]), np.sum(mock_eff)
    )
    np.testing.assert_array_equal(
        np.sum(output_datasets[0]["geometric_function"]), np.sum(mock_gf)
    )
    test_data_path = write_cdf(output_datasets[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1c_45sensor-heliopset_20250415-repoint00001_v999.cdf"
    )
