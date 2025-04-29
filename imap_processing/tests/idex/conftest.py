from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.idex.idex_l1a import PacketParser
from imap_processing.idex.idex_l1b import idex_l1b
from imap_processing.idex.idex_l2a import idex_l2a

TEST_DATA_PATH = imap_module_directory / "tests" / "idex" / "test_data"

TEST_L0_FILE_SCI = TEST_DATA_PATH / "imap_idex_l0_raw_20231218_v001.pkts"
TEST_L0_FILE_EVT = TEST_DATA_PATH / "imap_idex_l0_raw_20250108_v001.pkts"  # 1418
TEST_L0_FILE_CATLST = TEST_DATA_PATH / "imap_idex_l0_raw_20241206_v001.pkts"  # 1419

L1A_EXAMPLE_FILE = TEST_DATA_PATH / "idex_l1a_validation_file.h5"
L1B_EXAMPLE_FILE = TEST_DATA_PATH / "idex_l1b_validation_file.h5"

pytestmark = pytest.mark.external_test_data

SPICE_ARRAYS = [
    "ephemeris_position_x",
    "ephemeris_position_y",
    "ephemeris_position_z",
    "ephemeris_velocity_x",
    "ephemeris_velocity_y",
    "ephemeris_velocity_z",
    "right_ascension",
    "declination",
    "solar_longitude",
    "spin_phase",
]


@pytest.fixture
def decom_test_data_sci() -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xarray.Dataset
        A ``xarray`` dataset containing the science test data
    """
    return PacketParser(TEST_L0_FILE_SCI).data[0]


@pytest.fixture
def decom_test_data_catlst() -> xr.Dataset:
    """Return a ``xarray`` dataset containing the catalog list summary data.

    Returns
    -------
    dataset : xarray.Dataset
        A ``xarray`` dataset containing the catalog list summary data.
    """
    return PacketParser(TEST_L0_FILE_CATLST).data[0]


@pytest.fixture
def decom_test_data_evt() -> xr.Dataset:
    """Return a ``xarray`` dataset containing the event log data.

    Returns
    -------
    dataset : xarray.Dataset
        A ``xarray`` dataset containing the event log data.
    """
    return PacketParser(TEST_L0_FILE_EVT).data[0]


@pytest.fixture
def l1a_example_data(_download_test_data):
    """
    Pytest fixture to load example L1A data (produced by the IDEX team) for testing.

    Returns
    -------
    dict
      A dictionary containing the 6 waveform and telemetry arrays
    """
    return load_hdf_file(L1A_EXAMPLE_FILE)


@pytest.fixture
def l2a_dataset(decom_test_data_sci: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    spin_phase_angles = xr.DataArray(
        np.random.randint(0, 360, len(decom_test_data_sci.epoch))
    )
    with mock.patch(
        "imap_processing.idex.idex_l1b.get_spice_data",
        return_value={"spin_phase": spin_phase_angles},
    ):
        dataset = idex_l2a(idex_l1b(decom_test_data_sci))
    return dataset


@pytest.fixture
def l1b_example_data(_download_test_data):
    """
    Pytest fixture to load example L1B data (produced by the IDEX team) for testing.

    Returns
    -------
    dict
      A dictionary containing the 6 waveform and telemetry arrays
    """
    return load_hdf_file(L1B_EXAMPLE_FILE)


def get_spice_data_side_effect_func(l1a_ds, idex_attrs):
    # Create a mock dictionary of spice arrays

    return {
        name: xr.DataArray(
            name=name,
            data=np.ones(len(l1a_ds["epoch"])),
            dims="epoch",
            attrs=idex_attrs.get_variable_attributes(name),
        )
        for name in SPICE_ARRAYS
    }


def load_hdf_file(path: str) -> xr.Dataset:
    """
    Loads an HDF5 file produced by the IDEX team into a dataset.

    Parameters
    ----------
    path : str
        The file path to the HDF5 file.

    Returns
    -------
    dataset
        A dataset containing the extracted data.
    """
    # Load hdf5 data into a datatree
    datatree = xr.open_datatree(path, engine="netcdf4")
    datasets = []
    # Sort datatree by the event number
    datatree = sorted(datatree.items(), key=lambda x: int(x[0]))
    # Iterate through every nested tree in the datatree (Each nested tree represents
    # data from one event).
    # Rename the dimensions across every tree to be the same
    # Add an "event" dimension which will allow them all to be concatenated together.
    for event, tree in datatree:
        event_num = int(event)
        # Extract the metadata
        metadata = tree.Metadata.to_dataset().expand_dims({"event": [event_num]})
        ds = tree.to_dataset()
        # Sort dimensions by shape. The high sampling time dimension is always less
        # than the low sampling time.
        dims = [k for k, v in sorted(ds.sizes.items(), key=lambda item: item[1])]
        ds = ds.rename({dims[0]: "time_low", dims[1]: "time_high"}).expand_dims(
            {"event": [event_num]}
        )
        datasets.append(xr.merge([ds, metadata]))

    example_dataset = xr.concat(datasets, dim="event")

    return example_dataset
