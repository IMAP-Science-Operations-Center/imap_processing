import numpy as np
import pytest
import xarray as xr

from imap_processing.ultra.l1b.ultra_l1b import create_dataset, ultra_l1b


@pytest.fixture()
def mock_data_l1a_dict():
    # Create sample data for the xarray Dataset
    epoch = np.arange(
        "2024-02-07T15:28:37", "2024-02-07T16:24:50", dtype="datetime64[s]"
    )[:5]

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
def mock_data_l1b_dict():
    epoch = np.array(
        [760591786368000000, 760591787368000000, 760591788368000000],
        dtype="datetime64[ns]",
    )
    data_dict = {"epoch": epoch, "x_front": np.zeros(3), "y_front": np.zeros(3)}
    return data_dict


def test_create_dataset(mock_data_l1b_dict):
    """Tests that dataset is created as expected."""
    dataset = create_dataset(mock_data_l1b_dict, "imap_ultra_l1b_45sensor-de")

    assert "epoch" in dataset.coords
    assert dataset.coords["epoch"].dtype == "datetime64[ns]"
    assert dataset.attrs["Logical_source"] == "imap_ultra_l1b_45sensor-de"
    assert dataset["x_front"].attrs["UNITS"] == "mm"
    np.testing.assert_array_equal(dataset["x_front"], np.zeros(3))


def test_ultra_l1b(mock_data_l1a_dict):
    """Tests that L1b data is created."""
    output_datasets = ultra_l1b(mock_data_l1a_dict)

    assert len(output_datasets) == 3
    assert (
        output_datasets[0].attrs["Logical_source"]
        == "imap_ultra_l1b_45sensor-extendedspin"
    )
    assert (
        output_datasets[1].attrs["Logical_source"]
        == "imap_ultra_l1b_45sensor-cullingmask"
    )
    assert (
        output_datasets[2].attrs["Logical_source"] == "imap_ultra_l1b_45sensor-badtimes"
    )
    assert (
        output_datasets[0].attrs["Logical_source_description"]
        == "IMAP-Ultra Instrument Level-1B Extended Spin Data."
    )


def test_ultra_l1b_error(mock_data_l1a_dict):
    """Tests that L1a data throws an error."""
    data_dict = mock_data_l1a_dict.copy()
    data_dict["bad_key"] = data_dict.pop("imap_ultra_l1a_45sensor-rates")
    with pytest.raises(
        ValueError, match="Data dictionary does not contain the expected keys."
    ):
        ultra_l1b(data_dict)
