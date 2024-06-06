import numpy as np
import pytest
import xarray as xr

from imap_processing.ultra.l1b.ultra_l1b import create_dataset, ultra_l1b


@pytest.fixture()
def mock_data_dict():
    epoch = np.array(
        [760591786368000000, 760591787368000000, 760591788368000000],
        dtype="datetime64[ns]",
    )
    data_dict = {"imap_ultra_l1a_45sensor-de": xr.Dataset(coords={"epoch": epoch})}
    return data_dict


def test_create_dataset(mock_data_dict):
    """Tests that dataset is created as expected."""
    dataset = create_dataset(mock_data_dict, "imap_ultra_l1b_45sensor-de")

    assert "epoch" in dataset.coords
    assert dataset.coords["epoch"].dtype == "datetime64[ns]"
    assert dataset.attrs["Logical_source"] == "imap_ultra_l1b_45sensor-de"
    assert dataset["x_front"].attrs["UNITS"] == "mm"
    np.testing.assert_array_equal(dataset["x_front"], np.zeros(3))


def test_ultra_l1b(mock_data_dict):
    """Tests that L1b data is created."""
    output_datasets = ultra_l1b(mock_data_dict)

    assert len(output_datasets) == 1
    assert output_datasets[0].attrs["Logical_source"] == "imap_ultra_l1b_45sensor-de"
    assert (
        output_datasets[0].attrs["Logical_source_description"]
        == "IMAP-Ultra Instrument Level-1B Annotated Event Data."
    )
