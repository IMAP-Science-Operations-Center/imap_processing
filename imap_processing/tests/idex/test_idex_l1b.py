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
    get_spice_data,
    get_trigger_mode_and_level,
    idex_l1b,
    unpack_instrument_settings,
)
from imap_processing.tests.idex import conftest
from imap_processing.tests.idex.conftest import get_spice_data_side_effect_func


@pytest.fixture
@mock.patch("imap_processing.idex.idex_l1b.get_spice_data")
def l1b_dataset(mock_get_spice_data, decom_test_data_sci: xr.Dataset) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """

    mock_get_spice_data.side_effect = get_spice_data_side_effect_func
    dataset = idex_l1b(decom_test_data_sci)
    return dataset


@pytest.fixture
def mock_spice_functions():
    """Mock spice functions to avoid loading kernels."""
    with (
        mock.patch("imap_processing.idex.idex_l1b.imap_state") as mock_state,
        mock.patch(
            "imap_processing.idex.idex_l1b.instrument_pointing"
        ) as mock_pointing,
        mock.patch("imap_processing.idex.idex_l1b.solar_longitude") as mock_lon,
    ):
        mock_state.side_effect = lambda t, observer: np.ones((len(t), 6))
        mock_pointing.side_effect = lambda t, instrument, to_frame, cartesian: np.ones(
            (len(t), 3)
        )
        mock_lon.side_effect = lambda t, degrees: np.ones(len(t))

        yield mock_state, mock_pointing, mock_lon


def test_l1b_logical_source(l1b_dataset: xr.Dataset):
    """Tests that the ``idex_l1b`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    l1b_dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    expected_src = "imap_idex_l1b_sci-1week"
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
    assert file_name.name == "imap_idex_l1b_sci-1week_20231218_v999.cdf"


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
        assert l1b_dataset[var_name].attrs["UNITS"] == row["unit"]

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


def test_get_trigger_settings_success(decom_test_data_sci):
    """
    Check that the output to 'get_trigger_mode_and_level' matches expected arrays.

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        L1a dataset
    """
    # Change the trigger mode and level for the first event to check that output is
    # correct when the modes and levels vary from event to event
    decom_test_data_sci["idx__txhdrmgtrigmode"][0] = 1
    decom_test_data_sci["idx__txhdrhgtrigmode"][0] = 0

    n_epochs = len(decom_test_data_sci["epoch"])
    trigger_settings = get_trigger_mode_and_level(decom_test_data_sci)

    expected_modes = np.full(n_epochs, "HGThreshold")
    expected_modes[0] = "MGThreshold"
    expected_levels = np.full(n_epochs, 0.16762)
    expected_levels[0] = 1023.0 * 1.13e-2

    assert (trigger_settings["triggermode"].data == expected_modes).all(), (
        f"The dict entry 'triggermode' values did not match the expected values: "
        f"{expected_modes}. Found: {trigger_settings['triggermode'].data}"
    )

    assert (trigger_settings["triggerlevel"].data == expected_levels).all(), (
        f"The dict entry 'triggerlevel' values did not match the expected values: "
        f"{expected_levels}. Found: {trigger_settings['triggerlevel'].data}"
    )


def test_get_trigger_settings_failure(decom_test_data_sci):
    """
    Check that an error is thrown when there are more than one valid trigger for an
    event

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        L1a dataset
    """
    decom_test_data_sci["idx__txhdrhgtrigmode"][0] = 1
    decom_test_data_sci["idx__txhdrmgtrigmode"][0] = 2

    error_ms = (
        "Only one channel can trigger a dust event. Please make sure there is "
        "only one valid trigger value per event. This caused Merge Error: "
        "conflicting values for variable 'trigger_mode' on objects to be "
        "combined. You can skip this check by specifying compat='override'."
    )

    with pytest.raises(ValueError, match=error_ms):
        get_trigger_mode_and_level(decom_test_data_sci)


@pytest.mark.usefixtures("use_fake_spin_data_for_time")
def test_get_spice_data(
    mock_spice_functions,
    use_fake_spin_data_for_time,
    decom_test_data_sci,
    furnish_kernels,
):
    """
    Test the get_spice_data() function.

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        L1a dataset
    """
    kernels = ["naif0012.tls"]
    times = decom_test_data_sci["shcoarse"].data
    use_fake_spin_data_for_time(np.min(times), np.max(times))

    # Mock attribute manager variable attrs
    idex_attrs = ImapCdfAttributes()

    with (
        furnish_kernels(kernels),
        mock.patch.object(idex_attrs, "get_variable_attributes") as mock_attrs,
    ):
        mock_attrs.return_value = {"CATDESC": "Test var"}

        spice_data = get_spice_data(decom_test_data_sci, idex_attrs)

    for array in conftest.SPICE_ARRAYS:
        assert array in spice_data
        assert len(spice_data[array]) == len(decom_test_data_sci["epoch"])


@pytest.mark.external_test_data
def test_validate_l1b_idex_data_variables(
    l1b_dataset: xr.Dataset, l1b_example_data: xr.Dataset
):
    """
    Verify that each of the 6 waveform and telemetry arrays are equal to the
    corresponding array produced by the IDEX team using the same l0 file.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        The dataset to test with
    l1b_example_data: xr.Dataset
        A dataset containing the 6 waveform and telemetry arrays
    """
    # Lookup table to match the SDC array names to the Idex Team array names

    match_variables = {
        "TOF L": "TOF_Low",
        "TOF H": "TOF_High",
        "TOF M": "TOF_Mid",
        "Target H": "Target_High",
        "Target L": "Target_Low",
        "Ion Grid": "Ion_Grid",
        "Time (high sampling)": "time_high_sample_rate",
        "Time (low sampling)": "time_low_sample_rate",
        "current_2V5_bus": "current_2p5v_bus",
        "current_3V3_bus": "current_3p3v_bus",
        "current_neg2V5_bus": "current_neg2p5v_bus",
        "voltage_3V3_op_ref": "voltage_3p3_op_ref",
        "voltage_3V3_ref": "voltage_3p3_ref",
        "voltage_pos3V3_bus": "voltage_pos3p3v_bus",
    }

    # The Engineering data is converting to UTC, and the SDC is converting to J2000,
    # for 'epoch' and 'Timestamp' so this test is using the raw time value 'SCHOARSE' to
    # validate time
    # SPICE data is mocked.
    arrays_to_skip = [
        "Timestamp",
        "Epoch",
        "Pitch",
        "Roll",
        "Yaw",
        "Declination",
        "PositionX",
        "PositionY",
        "PositionZ",
        "VelocityX",
        "VelocityY",
        "VelocityZ",
        "RightAscension",
    ]
    # Compare each corresponding variable
    for var in l1b_example_data.data_vars:
        if var not in arrays_to_skip:
            # Get the corresponding array name
            cdf_var = match_variables.get(var, var.lower().replace(".", "p"))

            warning = (
                f"The array '{cdf_var}' does not equal the expected example array "
            )
            f"'{var}' produced by the IDEX team"

            if l1b_dataset[cdf_var].dtype == object:
                assert (l1b_dataset[cdf_var].data == l1b_example_data[var]).all(), (
                    warning
                )

            else:
                (
                    np.testing.assert_array_almost_equal(
                        l1b_dataset[cdf_var].data,
                        l1b_example_data[var],
                        decimal=1e-04,
                    ),
                    warning,
                )
