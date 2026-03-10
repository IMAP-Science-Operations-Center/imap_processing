from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.quality_flags import ImapDEOutliersUltraFlags
from imap_processing.ultra.l1b.de import FILLVAL_FLOAT32
from imap_processing.ultra.l1b.ultra_l1b import ultra_l1b
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.fixture
def mock_data_l1a_rates_dict():
    # Create sample data for the xarray Dataset
    epoch = np.arange(
        "2024-02-07T15:28:37", "2024-02-07T15:28:42", dtype="datetime64[s]"
    ).astype("datetime64[ns]")

    data_vars = {
        "COIN_TYPE": ("epoch", np.zeros(5)),
    }

    attrs = {
        "Logical_source": "imap_ultra_l1a_45sensor-rates",
        "Logical_source_description": "IMAP Mission ULTRA Instrument "
        "Level-1A Single-Sensor Data",
    }

    dataset = xr.Dataset(data_vars, coords={"epoch": epoch}, attrs=attrs)

    data_dict = {"imap_ultra_l1a_45sensor-rates": dataset}
    return data_dict


@pytest.fixture
def mock_data_l1b_de_dict():
    epoch = np.array(
        [760591786368000000, 760591787368000000, 760591788368000000],
        dtype="datetime64[ns]",
    )
    data_dict = {"epoch": epoch, "x_front": np.zeros(3), "y_front": np.zeros(3)}
    return data_dict


@pytest.fixture
def mock_data_l1b_extendedspin_dict():
    epoch = np.array(
        [760591786368000000, 760591787368000000, 760591788368000000],
        dtype="datetime64[ns]",
    )
    spin = np.array(
        [0, 1, 2],
        dtype="uint32",
    )
    energy = np.array(
        [0, 1],
        dtype="int32",
    )
    spin_start_time = np.array([0, 1, 2], dtype="uint64")
    quality = np.zeros((2, 3), dtype="uint16")
    # These should be shape: (3,)
    energy_dep_flags = np.zeros(len(spin), dtype="uint16")
    data_dict = {
        "epoch": epoch,
        "spin_number": spin,
        "energy_bin_geometric_mean": energy,
        "spin_start_time": spin_start_time,
        "quality_ena_rates": quality,
        "quality_low_voltage": energy_dep_flags,
        "quality_high_energy": energy_dep_flags,
        "quality_statistics": energy_dep_flags,
        "energy_range_flags": np.ones(5, dtype=np.uint16),
        "energy_range_edges": np.ones(4, dtype=np.float64),
    }
    return data_dict


@pytest.fixture
def mock_get_annotated_particle_velocity():
    """
    Mock behavior of get_annotated_particle_velocity.

    Returns NaN-filled arrays matching the expected output shape.
    """

    def side_effect_func(event_times, position, ultra_frame, dps_frame, sc_frame):
        num_events = event_times.size
        return (
            np.full((num_events, 3), np.nan),  # sc_velocity
            np.full((num_events, 3), np.nan),  # sc_dps_velocity
            np.full((num_events, 3), np.nan),  # helio_velocity
        )

    with mock.patch(
        "imap_processing.ultra.l1b.de.get_annotated_particle_velocity"
    ) as mocked_func:
        mocked_func.side_effect = side_effect_func
        yield mocked_func


def test_create_extendedspin_dataset(mock_data_l1b_extendedspin_dict):
    """Tests that dataset is created as expected."""
    dataset = create_dataset(
        mock_data_l1b_extendedspin_dict,
        "imap_ultra_l1b_45sensor-extendedspin",
        "l1b",
    )

    assert "spin_number" in dataset.coords
    assert "energy_bin_geometric_mean" in dataset.coords
    assert dataset.coords["spin_number"].dtype == "uint32"
    assert dataset.attrs["Logical_source"] == "imap_ultra_l1b_45sensor-extendedspin"
    assert dataset["quality_ena_rates"].attrs["UNITS"] == " "
    np.testing.assert_array_equal(
        dataset["quality_ena_rates"], np.zeros((2, 3), dtype="uint16")
    )


def test_create_de_dataset(mock_data_l1b_de_dict):
    """Tests that dataset is created as expected."""
    dataset = create_dataset(mock_data_l1b_de_dict, "imap_ultra_l1b_45sensor-de", "l1b")

    assert "epoch" in dataset.coords
    assert dataset.coords["epoch"].dtype == "datetime64[ns]"
    assert dataset.attrs["Logical_source"] == "imap_ultra_l1b_45sensor-de"
    assert dataset["x_front"].attrs["UNITS"] == "mm / 100"
    np.testing.assert_array_equal(dataset["x_front"], np.zeros(3))


@pytest.mark.external_test_data
def test_cdf_de(
    de_dataset,
    aux_dataset,
    use_fake_spin_data_for_time,
    ancillary_files,
    use_fake_repoint_data_for_time,
    mock_get_annotated_particle_velocity,
):
    """Tests that CDF file is created and contains same attributes as xarray."""

    data_dict = {}
    de_dataset.attrs["Repointing"] = "repoint00001"
    data_dict[de_dataset.attrs["Logical_source"]] = de_dataset
    data_dict[aux_dataset.attrs["Logical_source"]] = aux_dataset
    # Create a spin table that cover spin 0-141
    use_fake_spin_data_for_time(511000000, 511000000 + 86400 * 5)
    use_fake_repoint_data_for_time(np.arange(511000000, 511000000 + 86400 * 5, 86400))

    l1b_de_dataset = ultra_l1b(data_dict, ancillary_files)

    assert (
        l1b_de_dataset[0].attrs["Logical_source_description"]
        == "IMAP-Ultra Instrument Level-1B Direct Event Data."
    )

    l1b_de_dataset[0].attrs["Data_version"] = "999"
    l1b_de_dataset[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(l1b_de_dataset[0], istp=True)
    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1b_45sensor-de_20240207-repoint99999_v999.cdf"
    )
    # check that event_id exists in the dataset
    assert "event_id" in l1b_de_dataset[0].variables


@pytest.mark.external_test_data
def test_cdf_de_flags(
    mock_get_annotated_particle_velocity,
    de_dataset,
    aux_dataset,
    use_fake_spin_data_for_time,
    ancillary_files,
    use_fake_repoint_data_for_time,
):
    """Tests that the de code flags events not in a repointing."""
    data_dict = {}
    de_dataset.attrs["Repointing"] = "repoint00000"
    data_dict[de_dataset.attrs["Logical_source"]] = de_dataset
    data_dict[aux_dataset.attrs["Logical_source"]] = aux_dataset
    # Create a spin table that cover spin 0-141
    use_fake_spin_data_for_time(511000000, 511000000 + 86400 * 5)
    # Use repoint data that will NOT cover the event times to test flag setting
    use_fake_repoint_data_for_time(np.arange(0, +86400 * 5, 86400))

    l1b_de_dataset = ultra_l1b(data_dict, ancillary_files)
    # All valid events should be flagged as DURINGREPOINT since the repoint data does
    # not cover any of the event times
    valid_events = l1b_de_dataset[0]["event_times"] != FILLVAL_FLOAT32
    flags = l1b_de_dataset[0]["quality_outliers"].values[valid_events]
    assert np.all((flags & ImapDEOutliersUltraFlags.DURINGREPOINT.value) != 0)


@mock.patch("imap_processing.ultra.l1b.extendedspin.UltraConstants.SPIN_BIN_SIZE", 5)
@pytest.mark.external_test_data
def test_ultra_l1b_extendedspin(
    use_fake_spin_data_for_time, aux_dataset, rates_dataset, status_dataset
):
    """Tests that L1b data is created."""
    use_fake_spin_data_for_time(0, 141 * 15)
    l1b_de_dataset_path = (
        TEST_PATH / "imap_ultra_l1b_45sensor-de_20240207-repoint99999_v999.cdf"
    )
    l1b_de_dataset = load_cdf(l1b_de_dataset_path)
    data_dict = {
        key: l1b_de_dataset
        for key in [
            "imap_ultra_l1b_45sensor-de",
            "imap_ultra_l1a_45sensor-params",
        ]
    }
    data_dict["imap_ultra_l1a_45sensor-aux"] = aux_dataset
    data_dict["imap_ultra_l1a_45sensor-rates"] = rates_dataset
    data_dict["imap_ultra_l1b_45sensor-status"] = status_dataset

    ancillary_files = {}
    l1b_extendedspin_dataset = ultra_l1b(data_dict, ancillary_files)

    assert len(l1b_extendedspin_dataset) == 1
    assert (
        l1b_extendedspin_dataset[0].attrs["Logical_source"]
        == "imap_ultra_l1b_45sensor-extendedspin"
    )


@mock.patch("imap_processing.ultra.l1b.extendedspin.UltraConstants.SPIN_BIN_SIZE", 5)
@pytest.mark.external_test_data
def test_cdf_extendedspin(
    use_fake_spin_data_for_time, aux_dataset, rates_dataset, status_dataset
):
    use_fake_spin_data_for_time(0, 141 * 15)
    l1b_de_dataset_path = (
        TEST_PATH / "imap_ultra_l1b_45sensor-de_20240207-repoint99999_v999.cdf"
    )
    l1b_de_dataset = load_cdf(l1b_de_dataset_path)

    data_dict = {
        key: l1b_de_dataset
        for key in [
            "imap_ultra_l1b_45sensor-de",
            "imap_ultra_l1a_45sensor-params",
        ]
    }
    data_dict["imap_ultra_l1a_45sensor-aux"] = aux_dataset
    data_dict["imap_ultra_l1a_45sensor-rates"] = rates_dataset
    data_dict["imap_ultra_l1b_45sensor-status"] = status_dataset

    ancillary_files = {}
    l1b_extendedspin_dataset = ultra_l1b(data_dict, ancillary_files)
    """Tests that CDF file is created and contains same attributes as xarray."""
    l1b_extendedspin_dataset[0].attrs["Data_version"] = "999"
    l1b_extendedspin_dataset[0].attrs["Repointing"] = "repoint99999"
    l1b_extendedspin_dataset[0].attrs["Start_date"] = "20240207"
    test_data_path = write_cdf(l1b_extendedspin_dataset[0])
    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1b_45sensor-extendedspin_20240207-repoint99999_v999.cdf"
    )


@mock.patch("imap_processing.ultra.l1b.extendedspin.UltraConstants.SPIN_BIN_SIZE", 5)
@pytest.mark.external_test_data
def test_cdf_goodtimes(
    use_fake_spin_data_for_time, aux_dataset, rates_dataset, status_dataset
):
    """Tests that CDF file is created and contains same attributes as xarray."""
    use_fake_spin_data_for_time(0, 141 * 15)
    l1b_de_dataset_path = (
        TEST_PATH / "imap_ultra_l1b_45sensor-de_20240207-repoint99999_v999.cdf"
    )
    l1b_de_dataset = load_cdf(l1b_de_dataset_path)

    data_dict = {
        key: l1b_de_dataset
        for key in [
            "imap_ultra_l1b_45sensor-de",
            "imap_ultra_l1a_45sensor-params",
        ]
    }
    data_dict["imap_ultra_l1a_45sensor-aux"] = aux_dataset
    data_dict["imap_ultra_l1a_45sensor-rates"] = rates_dataset
    data_dict["imap_ultra_l1b_45sensor-status"] = status_dataset

    ancillary_files = {}
    l1b_extendedspin_dataset = ultra_l1b(data_dict, ancillary_files)

    goodtimes_dataset = ultra_l1b(
        {"imap_ultra_l1b_45sensor-extendedspin": l1b_extendedspin_dataset[0]},
        ancillary_files,
    )
    goodtimes_dataset[0].attrs["Data_version"] = "999"
    goodtimes_dataset[0].attrs["Repointing"] = "repoint99999"
    goodtimes_dataset[0].attrs["Start_date"] = "20240207"
    test_data_path = write_cdf(goodtimes_dataset[0])
    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1b_45sensor-goodtimes_20240207-repoint99999_v999.cdf"
    )


@mock.patch("imap_processing.ultra.l1b.extendedspin.UltraConstants.SPIN_BIN_SIZE", 5)
@pytest.mark.external_test_data
def test_cdf_badtimes(
    use_fake_spin_data_for_time, aux_dataset, rates_dataset, status_dataset
):
    """Tests that CDF file is created and contains same attributes as xarray."""
    use_fake_spin_data_for_time(0, 141 * 15)
    l1b_de_dataset_path = (
        TEST_PATH / "imap_ultra_l1b_45sensor-de_20240207-repoint99999_v999"
    )
    l1b_de_dataset = load_cdf(l1b_de_dataset_path)

    data_dict = {
        key: l1b_de_dataset
        for key in [
            "imap_ultra_l1b_45sensor-de",
            "imap_ultra_l1a_45sensor-params",
        ]
    }
    data_dict["imap_ultra_l1a_45sensor-aux"] = aux_dataset
    data_dict["imap_ultra_l1a_45sensor-rates"] = rates_dataset
    data_dict["imap_ultra_l1b_45sensor-status"] = status_dataset

    ancillary_files = {}
    l1b_extendedspin_dataset = ultra_l1b(data_dict, ancillary_files)

    ancillary_files = {}
    goodtimes_dataset = ultra_l1b(
        {"imap_ultra_l1b_45sensor-extendedspin": l1b_extendedspin_dataset[0]},
        ancillary_files,
    )

    l1b_badtimes_dataset = ultra_l1b(
        {
            "imap_ultra_l1b_45sensor-extendedspin": l1b_extendedspin_dataset[0],
            "imap_ultra_l1b_45sensor-goodtimes": goodtimes_dataset[0],
        },
        ancillary_files,
    )
    l1b_badtimes_dataset[0].attrs["Data_version"] = "999"
    l1b_badtimes_dataset[0].attrs["Repointing"] = "repoint99999"
    l1b_badtimes_dataset[0].attrs["Start_date"] = "20240207"
    test_data_path = write_cdf(l1b_badtimes_dataset[0])
    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1b_45sensor-badtimes_20240207-repoint99999_v999.cdf"
    )


def test_ultra_l1b_error(mock_data_l1a_rates_dict):
    """Tests that L1a data throws an error."""
    mock_data_l1a_rates_dict["bad_key"] = mock_data_l1a_rates_dict.pop(
        "imap_ultra_l1a_45sensor-rates"
    )
    ancillary_files = {}
    with pytest.raises(
        ValueError, match="Data dictionary does not contain the expected keys."
    ):
        ultra_l1b(mock_data_l1a_rates_dict, ancillary_files)
