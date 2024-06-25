import numpy as np
import pytest
import xarray as xr

from imap_processing.ultra.l1c.ultra_l1c import ultra_l1c
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset


@pytest.fixture()
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


@pytest.fixture()
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
    output_datasets = ultra_l1c(mock_data_l1b_dict, data_version="001")

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
    with pytest.raises(
        ValueError, match="Data dictionary does not contain the expected keys."
    ):
        ultra_l1c(mock_data_l1b_dict, data_version="001")
