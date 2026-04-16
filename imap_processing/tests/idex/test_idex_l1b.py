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
    TRIGGER_LABELS,
    EventMessage,
    TriggerOrigin,
    get_spice_data,
    get_trigger_mode_and_level,
    get_trigger_origin,
    idex_l1b,
    unpack_instrument_settings,
)
from imap_processing.idex.idex_utils import get_idex_attrs
from imap_processing.tests.idex import conftest


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
    idex_attrs = get_idex_attrs("l1b")
    n_epochs = len(decom_test_data_sci["epoch"])
    trigger_settings = get_trigger_mode_and_level(decom_test_data_sci, idex_attrs)

    expected_modes_lg = np.full(n_epochs, None)
    expected_modes_hg = expected_modes_lg.copy()
    expected_modes_hg[1:] = "HGThreshold"
    expected_modes_mg = np.full(n_epochs, None)
    expected_modes_mg[0] = "MGThreshold"
    expected_levels_lg = np.full(n_epochs, np.nan)
    expected_levels_hg = expected_levels_lg.copy()
    expected_levels_hg[1:] = 0.16762
    expected_levels_mg = expected_levels_lg.copy()
    expected_levels_mg[0] = 1023.0 * 1.13e-2

    var_names = ["trigger_mode_lg", "trigger_mode_mg", "trigger_mode_hg"]
    expected_modes = [expected_modes_lg, expected_modes_mg, expected_modes_hg]
    for expected_mode, mode_name in zip(expected_modes, var_names, strict=False):
        (
            np.testing.assert_array_equal(
                trigger_settings[mode_name].data,
                expected_mode,
                err_msg=f"The dict entry {mode_name} values did not match the"
                f" expected values: {expected_mode}. Found:"
                f" {trigger_settings[mode_name].data}",
            ),
        )
    var_names = ["trigger_level_lg", "trigger_level_mg", "trigger_level_hg"]
    expected_levels = [expected_levels_lg, expected_levels_mg, expected_levels_hg]
    for expected_level, level_name in zip(expected_levels, var_names, strict=False):
        (
            np.testing.assert_array_equal(
                trigger_settings[level_name].data,
                expected_level,
                err_msg=f"The dic entry {level_name} values did not match the"
                f" expected values: {expected_level}. Found: "
                f"{trigger_settings[level_name].data}",
            ),
        )


def test_trigger_origin():
    """Check that the correct labels are produced for trigger origin values"""

    trigger_bits = np.full(10, 6)
    origins = get_trigger_origin(trigger_bits, get_idex_attrs("l1b"))
    # Bits 1 and 2 should be set for all events
    expected_origin = np.full(
        10,
        ", ".join([TRIGGER_LABELS[TriggerOrigin(1)], TRIGGER_LABELS[TriggerOrigin(2)]]),
    )
    np.testing.assert_array_equal(
        origins["trigger_origin"],
        expected_origin,
        err_msg=f"The trigger origin values did not match the expected values: "
        f"{expected_origin}. Found: {origins}",
    )


def test_invalid_trigger_origin():
    """Check the labels when there are invalid trigger origin values"""

    trigger_bits = np.full(10, 64)  # invalid trigger origin values
    origins = get_trigger_origin(trigger_bits, get_idex_attrs("l1b"))
    # Bits 1 and 2 should be set for all events
    expected_origin = np.full(10, "Unknown trigger origin")
    np.testing.assert_array_equal(
        origins["trigger_origin"],
        expected_origin,
        err_msg=f"The trigger origin values did not match the expected values:"
        f"{expected_origin}. Found: {origins}",
    )


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
        "HGTriggerLevel": "trigger_level_hg",
        "MGTriggerLevel": "trigger_level_mg",
        "LGTriggerLevel": "trigger_level_lg",
        "TriggerOrigin": "trigger_origin",
    }

    # The Engineering data is converting to UTC, and the SDC is converting to J2000,
    # for 'epoch' and 'Timestamp' so this test is using the raw time value 'SCHOARSE' to
    # validate time
    # SPICE data is mocked.
    arrays_to_skip = [
        "Timestamp",
        "epoch",
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
        "FIFODelay",
        "FIFODelayMicroseconds",
        "FIFODelay_H",
        "FIFODelay_L",
        "FIFODelay_M",
        "HSPosttriggerBlocks",
    ]
    # assert that "aid" is in l1b
    assert "aid" in l1b_dataset, "The array 'aid' is missing from the l1b dataset."
    # select only the first n events
    l1b_example_data = l1b_example_data.isel(
        event=np.arange(l1b_dataset.sizes["epoch"])
    )
    # Compare each corresponding variable
    for var in l1b_example_data.data_vars:
        if var not in arrays_to_skip:
            # Get the corresponding array name
            cdf_var = match_variables.get(var, var.lower().replace(".", "p"))
            warning = (
                f"The array '{cdf_var}' does not equal the expected example array "
                f"'{var}' produced by the IDEX team"
            )
            # TODO remove this block once the IDEX team fixes the l1b validation file.
            #   They included a lot of extra variables in the current file.
            try:
                l1b_dataset[cdf_var]
            except KeyError:
                continue
            if l1b_dataset[cdf_var].dtype == object:
                assert (
                    l1b_dataset[cdf_var].data == np.squeeze(l1b_example_data[var])
                ).all(), warning

            else:
                np.testing.assert_array_almost_equal(
                    l1b_dataset[cdf_var].data,
                    np.squeeze(l1b_example_data[var]),
                    decimal=4,
                    err_msg=warning,
                )


def test_l1b_msg_processing(decom_test_data_msg: xr.Dataset):
    """Verify that the MSG data is being processed correctly in the l1b processing.

    Parameters
    ----------
    decom_test_data_msg : xr.Dataset
        A dataset containing the MSG data produced by the l1a processing.
    """
    msg_ds = decom_test_data_msg.copy()
    # Set 2 consecutive events to have pulser on and pulser off
    msg_ds.messages[2] = EventMessage.PULSER_ON.value
    msg_ds.messages[3] = EventMessage.PULSER_OFF.value
    # Set 2 to have a non-consecutive pulser on and pulser off to check that
    # non-consecutive events are treated as non-valid pulser on and off events
    msg_ds.messages[20] = EventMessage.PULSER_ON.value
    msg_ds.messages[22] = EventMessage.PULSER_OFF.value
    # Process the MSG data with the l1b function
    test_l1b_msg = idex_l1b(msg_ds, "msg")
    expected_vars = [
        "epoch",
        "pulser_on",
        "science_on",
    ]
    for var in expected_vars:
        assert var in test_l1b_msg, (
            f"The variable '{var}' is missing from the MSG dataset."
        )

    # Check that the pulser_on variable is correct
    expected_pulser_on = np.ones_like(test_l1b_msg["pulser_on"]) * 255
    # The pulser_on variable should be 1 for the 1st and 0 for the 2nd event, and
    # 255 for all other events
    expected_pulser_on[0] = 1
    expected_pulser_on[1] = 0
    # Check that the science_on variable is correct
    expected_science_on = np.ones_like(test_l1b_msg["pulser_on"]) * 255
    # The science_on variable should be 1 for the 3rd event and 0 for the 4th event
    expected_science_on[2] = 1
    expected_science_on[3] = 0
    np.testing.assert_array_equal(test_l1b_msg["pulser_on"].data, expected_pulser_on)
    np.testing.assert_array_equal(test_l1b_msg["science_on"].data, expected_science_on)


def test_no_valid_messages(decom_test_data_msg: xr.Dataset):
    """Verify that if there are no valid pulser and science events then no dataset is
    returned.

    Parameters
    ----------
    decom_test_data_msg : xr.Dataset
        A dataset containing the MSG data produced by the l1a processing.
    """
    msg_ds = decom_test_data_msg.copy()
    # Set all messages to a value that is not a valid pulser on or off event
    msg_ds.messages[:] = "Not a science or pulser event"
    result = idex_l1b(msg_ds, "msg")
    assert result is None
