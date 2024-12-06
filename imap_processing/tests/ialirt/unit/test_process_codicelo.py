import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.ialirt.l0.process_codicelo import (
    find_groups,
    process_codicelo,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_codicelo_path():
    """Returns the xtce directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_codicelo.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the xtce auxiliary directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "lo_fsw_view_0_ccsds.bin"
    )


@pytest.fixture(scope="session")
def codicelo_test_data():
    """Returns the xtce auxiliary directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "imap_codice_l1a_lo-ialirt_20240429164800_v0.0.0.cdf"
    )
    data = load_cdf(data_path)

    return data


import numpy as np
import pytest
import xarray as xr

@pytest.fixture()
def xarray_data(binary_packet_path, xtce_codicelo_path):
    """Create xarray data"""
    apid = 1152

    # Load the dataset using the provided function
    xarray_data = packet_file_to_datasets(binary_packet_path, xtce_codicelo_path)[apid]

    # Define the desired total number of epochs
    original_epochs = len(xarray_data['epoch'])
    num_new_values = 240  # Number of new epochs to add

    # Generate the new `src_seq_ctr` values with a repeating range 0 to 239
    total_epochs = original_epochs + num_new_values
    new_src_seq_ctr = np.arange(total_epochs) % 240
    new_src_seq_ctr = np.delete(new_src_seq_ctr, 400)
    new_src_seq_ctr = np.append(new_src_seq_ctr, 0)

    # Generate new unique `epoch` values
    max_existing_epoch = xarray_data['epoch'].values.max()
    new_epochs = np.arange(max_existing_epoch + 1, max_existing_epoch + 1 + num_new_values)

    # Concatenate `epoch` coordinate
    all_epochs = np.concatenate([xarray_data['epoch'].values, new_epochs])

    # Generate `cod_lo_acq` values dynamically
    original_cod_lo_acq = xarray_data['cod_lo_acq'].values
    new_cod_lo_acq = np.full(num_new_values, original_cod_lo_acq[-1])
    new_cod_lo_acq[new_src_seq_ctr[original_epochs:] == 0] = 452105281
    cod_lo_acq_full = np.arange(452105280, 452105280 + 480, dtype=np.uint32)

    # Duplicate the other variables
    new_data_vars = {}
    for var_name in xarray_data.data_vars:
        if var_name == 'src_seq_ctr':
            new_data_vars[var_name] = xr.DataArray(new_src_seq_ctr, dims=["epoch"], coords={"epoch": all_epochs})
        elif var_name == 'cod_lo_acq':
            new_data_vars[var_name] = xr.DataArray(cod_lo_acq_full, dims=["epoch"], coords={"epoch": all_epochs})
        else:
            # Repeat the last value for other variables
            repeated_values = xarray_data[var_name].isel(epoch=-1).expand_dims(epoch=new_epochs)
            new_data_vars[var_name] = xr.concat([xarray_data[var_name], repeated_values], dim="epoch")

    # Create the updated dataset
    updated_dataset = xr.Dataset(new_data_vars, coords={"epoch": all_epochs})

    return updated_dataset


def test_find_groups(xarray_data):
    """Tests find_groups"""

    filtered_data = find_groups(xarray_data)

    np.testing.assert_array_equal(
        filtered_data["hit_subcom"], np.tile(np.arange(60), 15)
    )


def test_process_codicelo(xarray_data, codicelo_test_data):
    """Tests process_hit."""

    # Tests that it functions normally
    codicelo_product = process_codicelo(xarray_data)

    print('hi')



