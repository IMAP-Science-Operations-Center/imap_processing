import numpy as np
import pytest
import xarray as xr

from imap_processing import decom
from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.ultra.l0.decom_ultra import process_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
)
from imap_processing.ultra.l1a import ultra_l1a
from imap_processing.ultra.l1b.ultra_l1b import ultra_l1b
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset
from imap_processing.utils import group_by_apid


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
def mock_data_l1a_de_aux_dict():
    # Create sample data for the xarray Dataset
    epoch = np.arange(
        "2024-02-07T15:28:37", "2024-02-07T15:28:42", dtype="datetime64[s]"
    ).astype("datetime64[ns]")

    data_vars = {
        "var": ("epoch", np.zeros(5)),
    }

    attrs = {
        "Logical_source": "imap_ultra_l1a_45sensor-name",
        "Logical_source_description": "IMAP Mission ULTRA Instrument "
        "Level-1A Single-Sensor Data",
    }

    dataset = xr.Dataset(data_vars, coords={"epoch": epoch}, attrs=attrs)

    data_dict = {
        "imap_ultra_l1a_45sensor-de": dataset,
        "imap_ultra_l1a_45sensor-aux": dataset,
    }

    return data_dict


@pytest.fixture()
def mock_data_l1b_dict():
    epoch = np.array(
        [760591786368000000, 760591787368000000, 760591788368000000],
        dtype="datetime64[ns]",
    )
    data_dict = {"epoch": epoch, "x_front": np.zeros(3), "y_front": np.zeros(3)}
    return data_dict


@pytest.fixture()
def de_dataset(ccsds_path_theta_0, xtce_path):
    """L1A test data"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    decom_ultra_events = process_ultra_apids(
        grouped_data[ULTRA_EVENTS.apid[0]], ULTRA_EVENTS.apid[0]
    )
    decom_ultra_aux = process_ultra_apids(
        grouped_data[ULTRA_AUX.apid[0]], ULTRA_AUX.apid[0]
    )
    dataset = ultra_l1a.create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )
    # Remove start_type with fill values
    l1a_de_dataset = dataset.where(
        dataset["START_TYPE"] != GlobalConstants.INT_FILLVAL, drop=True
    )
    return l1a_de_dataset


def test_create_dataset(mock_data_l1b_dict):
    """Tests that dataset is created as expected."""
    dataset = create_dataset(mock_data_l1b_dict, "imap_ultra_l1b_45sensor-de", "l1b")

    assert "epoch" in dataset.coords
    assert dataset.coords["epoch"].dtype == "datetime64[ns]"
    assert dataset.attrs["Logical_source"] == "imap_ultra_l1b_45sensor-de"
    assert dataset["x_front"].attrs["UNITS"] == "hundredths of a millimeter"
    np.testing.assert_array_equal(dataset["x_front"], np.zeros(3))


def test_ultra_l1b_rates(mock_data_l1a_rates_dict):
    """Tests that L1b data is created."""
    output_datasets = ultra_l1b(mock_data_l1a_rates_dict, data_version="001")

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


def test_ultra_l1b_de(de_dataset):
    """Tests that L1b data is created."""
    data_dict = {}
    data_dict[de_dataset.attrs["Logical_source"]] = de_dataset
    data_dict["imap_ultra_l1a_45sensor-aux"] = de_dataset
    output_datasets = ultra_l1b(data_dict, data_version="001")

    assert len(output_datasets) == 1
    assert output_datasets[0].attrs["Logical_source"] == "imap_ultra_l1b_45sensor-de"
    assert (
        output_datasets[0].attrs["Logical_source_description"]
        == "IMAP-Ultra Instrument Level-1B Direct Event Data."
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
