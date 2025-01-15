"""Tests the L1b processing for IDEX data"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import write_cdf
from imap_processing.idex.idex_l1b import (
    get_trigger_mode_and_level,
    idex_l1b,
    unpack_instrument_settings,
)


@pytest.fixture(scope="module")
def l1b_dataset(decom_test_data: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    dataset = idex_l1b(decom_test_data, data_version="001")
    return dataset


def test_l1b_cdf_filenames(l1b_dataset: xr.Dataset):
    """Tests that the ``idex_l1b`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    l1b_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_src = "imap_idex_l1b_sci"
    assert l1b_dataset.attrs["Logical_source"] == expected_src


def test_idex_cdf_file(l1b_dataset: xr.Dataset):
    """Verify the CDF file can be created with no errors.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The dataset to test with
    """

    file_name = write_cdf(l1b_dataset)

    assert file_name.exists()
    assert file_name.name == "imap_idex_l1b_sci_20231214_v001.cdf"


def test_idex_waveform_units(l1b_dataset: xr.Dataset):
    """Verify the CDF instrument settings and waveforms have the correct units.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The dataset to test with
    """
    cdf_var_defs_path = (
        f"{imap_module_directory}/idex/idex_variable_unpacking_and_eu_conversion.csv"
    )
    cdf_var_defs = pd.read_csv(cdf_var_defs_path)

    # Check instrument setting units
    for _, row in cdf_var_defs.iterrows():
        var_name = row["mnemonic"]
        assert l1b_dataset[var_name].attrs["units"] == row["unit"]

    # Check waveform units
    waveform_var_names = [
        "TOF_High",
        "TOF_Low",
        "TOF_Mid",
        "Ion_Grid",
        "Target_Low",
        "Target_High",
    ]

    for var_name in waveform_var_names:
        assert l1b_dataset[var_name].attrs["UNITS"] == "pC"


def test_unpack_instrument_settings():
    """
    Check that the instrument setting variables are being unpacked correctly

    Example
    -------
    In this example, we are using a test variable that has five bits
    Idx__test_var01 = 0b10010

    Int(0b10010) = 18

    This should unpack into test_var0, and test_var1
     - test_var0 is two bits long and starts at 0, and the unpacked value should be 2
     - test_var1 is three bits long and starts at 3, and the unpacked value should be 4
    """
    # Create test dataset with an array shape = 5 all values = 18
    test_ds = xr.Dataset({"idx__test_var01": xr.DataArray(np.full(5, 18))})

    test_cdf_defs_df = pd.DataFrame(
        {
            "mnemonic": ["test_var0", "test_var1"],
            "var_name": ["idx__test_var01", "idx__test_var01"],
            "starting_bit": [0, 2],
            "nbits_padding_before": [0, 0],
            "unsigned_nbits": [2, 3],
        }
    )
    idex_attrs = ImapCdfAttributes()
    # Mock attribute manager variable attrs
    with mock.patch.object(
        idex_attrs, "get_variable_attributes", return_value={"CATDESC": "Test var"}
    ):
        unpacked_dict = unpack_instrument_settings(
            test_ds, test_cdf_defs_df, idex_attrs
        )

    assert np.all(unpacked_dict["test_var0"] == 2)
    assert np.all(unpacked_dict["test_var1"] == 4)


def test_get_trigger_settings_success(decom_test_data):
    """
    Check that the output to 'get_trigger_mode_and_level' matches expected arrays.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        L1a dataset
    """
    # Change the trigger mode and level for the first event to check that output is
    # correct when the modes and levels vary from event to event
    decom_test_data["idx__txhdrmgtrigmode"][0] = 1
    decom_test_data["idx__txhdrhgtrigmode"][0] = 0

    n_epochs = len(decom_test_data["epoch"])
    trigger_settings = get_trigger_mode_and_level(decom_test_data)

    expected_modes = np.full(n_epochs, "HGThreshold")
    expected_modes[0] = "MGThreshold"
    expected_levels = np.full(n_epochs, 0.16762)
    expected_levels[0] = 1023.0

    assert (trigger_settings["triggermode"].data == expected_modes).all(), (
        f"The dict entry 'triggermode' values did not match the expected values: "
        f"{expected_modes}. Found: {trigger_settings['triggermode'].data}"
    )

    assert (trigger_settings["triggerlevel"].data == expected_levels).all(), (
        f"The dict entry 'triggerlevel' values did not match the expected values: "
        f"{expected_levels}. Found: {trigger_settings['triggerlevels'].data}"
    )


def test_get_trigger_settings_failure(decom_test_data):
    """
    Check that an error is thrown when there are more than one valid trigger for an
    event

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        L1a dataset
    """
    decom_test_data["idx__txhdrhgtrigmode"][0] = 1
    decom_test_data["idx__txhdrmgtrigmode"][0] = 2

    error_ms = (
        "Only one channel can trigger a dust event. Please make sure there is "
        "only one valid trigger value per event. This caused Merge Error: "
        "conflicting values for variable 'trigger_mode' on objects to be "
        "combined. You can skip this check by specifying compat='override'."
    )

    with pytest.raises(ValueError, match=error_ms):
        get_trigger_mode_and_level(decom_test_data)
