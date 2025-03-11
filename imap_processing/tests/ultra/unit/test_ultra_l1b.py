import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import write_cdf
from imap_processing.ultra.l1b.ultra_l1b import ultra_l1b
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


@pytest.fixture()
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


@pytest.fixture()
def mock_data_l1b_de_dict():
    epoch = np.array(
        [760591786368000000, 760591787368000000, 760591788368000000],
        dtype="datetime64[ns]",
    )
    data_dict = {"epoch": epoch, "x_front": np.zeros(3), "y_front": np.zeros(3)}
    return data_dict


@pytest.fixture()
def mock_data_l1b_extendedspin_dict():
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
    data_dict = {
        "spin_number": spin,
        "energy_bin_geometric_mean": energy,
        "spin_start_time": spin_start_time,
        "quality_ena_rates": quality,
    }
    return data_dict


def test_create_extendedspin_dataset(mock_data_l1b_extendedspin_dict):
    """Tests that dataset is created as expected."""
    dataset = create_dataset(
        mock_data_l1b_extendedspin_dict,
        "imap_ultra_l1b_45sensor-extendedspin",
        "l1b",
        "001",
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
    dataset = create_dataset(
        mock_data_l1b_de_dict, "imap_ultra_l1b_45sensor-de", "l1b", "001"
    )

    assert "epoch" in dataset.coords
    assert dataset.coords["epoch"].dtype == "datetime64[ns]"
    assert dataset.attrs["Logical_source"] == "imap_ultra_l1b_45sensor-de"
    assert dataset["x_front"].attrs["UNITS"] == "mm / 100"
    np.testing.assert_array_equal(dataset["x_front"], np.zeros(3))


def test_ultra_l1b(l1b_de_dataset):
    """Tests that L1b data is created."""

    assert len(l1b_de_dataset) == 1

    assert (
        l1b_de_dataset[0].attrs["Logical_source_description"]
        == "IMAP-Ultra Instrument Level-1B Direct Event Data."
    )


def test_cdf_de(l1b_de_dataset):
    """Tests that CDF file is created and contains same attributes as xarray."""
    test_data_path = write_cdf(l1b_de_dataset[0], istp=False)
    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1b_45sensor-de_20240207_v001.cdf"


def test_ultra_l1b_extendedspin(l1b_extendedspin_dataset):
    """Tests that L1b data is created."""

    assert len(l1b_extendedspin_dataset) == 3

    # Define the suffixes and prefix
    prefix = "imap_ultra_l1b_45sensor"
    suffixes = ["extendedspin", "cullingmask", "badtimes"]

    for i in range(len(suffixes)):
        expected_logical_source = f"{prefix}-{suffixes[i]}"
        assert (
            l1b_extendedspin_dataset[i].attrs["Logical_source"]
            == expected_logical_source
        )


def test_cdf_extendedspin(l1b_extendedspin_dataset):
    """Tests that CDF file is created and contains same attributes as xarray."""
    test_data_path = write_cdf(l1b_extendedspin_dataset[0], istp=False)
    assert test_data_path.exists()
    assert (
        test_data_path.name == "imap_ultra_l1b_45sensor-extendedspin_20000101_v001.cdf"
    )


def test_ultra_l1b_error(mock_data_l1a_rates_dict):
    """Tests that L1a data throws an error."""
    mock_data_l1a_rates_dict["bad_key"] = mock_data_l1a_rates_dict.pop(
        "imap_ultra_l1a_45sensor-rates"
    )
    with pytest.raises(
        ValueError, match="Data dictionary does not contain the expected keys."
    ):
        ultra_l1b(mock_data_l1a_rates_dict, data_version="001")
