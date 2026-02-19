import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from imap_processing.glows.l1b.glows_l1b import glows_l1b, glows_l1b_de
from imap_processing.glows.l1b.glows_l1b_data import (
    AncillaryParameters,
    DirectEventL1B,
    HistogramL1B,
    PipelineSettings,
)
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.tests.glows.conftest import mock_update_spice_parameters


def test_glows_l1b_ancillary_file():
    fake_good_input = {
        "version": "0.1",
        "filter_temperature": {
            "min": -30.0,
            "max": 80.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
        "hv_voltage": {
            "min": 0.0,
            "max": 3500.0,
            "n_bits": 12,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
        "spin_period": {"min": 0.0, "max": 20.9712, "n_bits": 16},
        "spin_phase": {"min": 0.0, "max": 360.0, "n_bits": 16},
        "pulse_length": {
            "min": 0.0,
            "max": 255.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
    }

    ancillary = AncillaryParameters(fake_good_input)
    for key, data in fake_good_input.items():
        assert getattr(ancillary, key) == data

    fake_bad_input = {
        "version": "0.1",
        "filter_temperature": {
            "min": -30.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
    }

    with pytest.raises(KeyError):
        ancillary = AncillaryParameters(fake_bad_input)


def test_glows_l1b_de():
    input_test_data = np.array([[1, 0, 3], [100, 2_000, 6]])
    times, pulse_len = DirectEventL1B.process_direct_events(input_test_data)

    expected_times = np.array([1.0, 100.001])

    expected_pulse = np.array([3, 6])

    assert np.allclose(times, expected_times)
    assert np.allclose(pulse_len, expected_pulse)


@patch.object(HistogramL1B, "update_spice_parameters", autospec=True)
def test_validation_data_histogram(
    mock_spice_function,
    l1a_dataset,
    mock_ancillary_exclusions,
    mock_pipeline_settings,
    mock_conversion_table_dict,
):
    mock_spice_function.side_effect = mock_update_spice_parameters
    # Only test with histogram data (l1a_dataset[0])
    l1b = glows_l1b(
        l1a_dataset[0],
        mock_ancillary_exclusions.excluded_regions,
        mock_ancillary_exclusions.uv_sources,
        mock_ancillary_exclusions.suspected_transients,
        mock_ancillary_exclusions.exclusions_by_instr_team,
        mock_pipeline_settings,
        mock_conversion_table_dict,
    )
    end_time = l1b["epoch"].data[-1]

    validation_data = (
        Path(__file__).parent
        / "validation_data"
        / "imap_glows_l1b_hist_full_output.json"
    )
    with open(validation_data) as f:
        out = json.load(f)

    # TODO block header, flags
    expected_matching_columns = {
        "glows_start_time": "glows_start_time",
        "glows_end_time_offset": "glows_time_offset",
        "imap_start_time": "imap_start_time",
        "imap_end_time_offset": "imap_time_offset",
        "number_of_bins_per_histogram": "number_of_bins_per_histogram",
        "histogram": "histogram",
        "number_of_events": "number_of_events",
        # "imap_spin_angle_bin_cntr": "imap_spin_angle_bin_cntr",
        # "histogram_flag_array": "histogram_flag_array",
        "filter_temperature_average": "filter_temperature_average",
        "filter_temperature_std_dev": "filter_temperature_std_dev",
        "hv_voltage_average": "hv_voltage_average",
        "hv_voltage_std_dev": "hv_voltage_std_dev",
        "spin_period_average": "spin_period_average",
        "spin_period_std_dev": "spin_period_std_dev",
        "pulse_length_average": "pulse_length_average",
        "pulse_length_std_dev": "pulse_length_std_dev",
        # TODO uncomment when spice is complete
        # "spin_period_ground_average": "spin_period_ground_average",
        # "spin_period_ground_std_dev": "spin_period_ground_std_dev",
        # "position_angle_offset_average": "position_angle_offset_average",
        # "position_angle_offset_std_dev": "position_angle_offset_std_dev",
        # "spin_axis_orientation_average": "spin_axis_orientation_average",
        # "spin_axis_orientation_std_dev": "spin_axis_orientation_std_dev",
        # "spacecraft_location_average": "spacecraft_location_average",
        # "spacecraft_location_std_dev": "spacecraft_location_std_dev",
        # "spacecraft_velocity_average": "spacecraft_velocity_average",
        # "spacecraft_velocity_std_dev": "spacecraft_velocity_std_dev",
    }

    for validation_output in out["output"]:
        epoch_val = met_to_ttj2000ns(validation_output["imap_start_time"])

        # Skip validation data that doesn't match our single dataset timerange
        if epoch_val > end_time:
            continue
        datapoint = l1b.sel(epoch=epoch_val)

        assert np.equal(
            validation_output["imap_start_time"],
            datapoint.imap_start_time.data,
        )

        for key in validation_output:
            if key not in expected_matching_columns.keys():
                continue
            np.testing.assert_array_almost_equal(
                datapoint[expected_matching_columns[key]].data,
                validation_output[key],
                decimal=1,
            )


def test_validation_data_de(
    l1a_dataset,
    mock_ancillary_exclusions,
    mock_pipeline_settings,
    mock_conversion_table_dict,
):
    de_data = l1a_dataset[1]

    l1b = glows_l1b_de(de_data, mock_conversion_table_dict)
    validation_data = (
        Path(__file__).parent / "validation_data" / "imap_glows_l1b_de_output.json"
    )
    with open(validation_data) as f:
        out = json.load(f)

    expected_matching_columns = {
        "imap_time_last_pps",
        "imap_time_next_pps",
        "glows_time_last_pps",
        "number_of_completed_spins",
        "filter_temperature",
        "hv_voltage",
        "spin_period",
        "spin_phase_at_next_pps",
        "direct_event_glows_times",
        "direct_event_pulse_lengths",
    }

    for index, validation_output in enumerate(out["output"]):
        for key in validation_output:
            if key not in expected_matching_columns:
                continue
            if key in ["direct_event_glows_times", "direct_event_pulse_lengths"]:
                validation_length = len(validation_output[key])
                np.testing.assert_array_almost_equal(
                    l1b[key].isel(epoch=index).data[:validation_length],
                    validation_output[key],
                    decimal=1,
                )
            else:
                np.testing.assert_array_almost_equal(
                    l1b[key].isel(epoch=index).data, validation_output[key], decimal=1
                )


@pytest.mark.parametrize(
    "flags, expected",
    [
        (0, np.zeros(10)),
        (64, np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])),
        (65, np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])),
    ],
)
def test_deserialize_flags(flags, expected):
    output = HistogramL1B.deserialize_flags(flags)
    assert np.array_equal(output, expected)


def test_pipeline_settings_from_flattened_json():
    """PipelineSettings correctly reads flags from flattened JSON format.

    convert_json_to_dataset flattens nested dicts, so
    active_bad_time_flags.is_night -> active_bad_time_flags_is_night.
    PipelineSettings must reconstruct the ordered flag lists from these keys.
    """
    data_vars = {
        "active_bad_time_flags_is_pps_missing": ([], True),
        "active_bad_time_flags_is_time_status_missing": ([], True),
        "active_bad_time_flags_is_phase_missing": ([], True),
        "active_bad_time_flags_is_spin_period_missing": ([], True),
        "active_bad_time_flags_is_overexposed": ([], True),
        "active_bad_time_flags_is_direct_event_non_monotonic": ([], True),
        "active_bad_time_flags_is_night": ([], False),
        "active_bad_time_flags_is_hv_test_in_progress": ([], True),
        "active_bad_time_flags_is_test_pulse_in_progress": ([], True),
        "active_bad_time_flags_is_memory_error_detected": ([], True),
        "active_bad_time_flags_is_generated_on_ground": ([], True),
        "active_bad_time_flags_is_beyond_daily_statistical_error": (
            [],
            True,
        ),
        "active_bad_time_flags_is_temperature_std_dev_beyond_threshold": (
            [],
            True,
        ),
        "active_bad_time_flags_is_hv_voltage_std_dev_beyond_threshold": (
            [],
            True,
        ),
        "active_bad_time_flags_is_spin_period_std_dev_beyond_threshold": (
            [],
            True,
        ),
        "active_bad_time_flags_is_pulse_length_std_dev_beyond_threshold": (
            [],
            True,
        ),
        "active_bad_time_flags_is_spin_period_difference_beyond_threshold": (
            [],
            False,
        ),
        "active_bad_angle_flags_is_close_to_uv_source": ([], True),
        "active_bad_angle_flags_is_inside_excluded_region": ([], True),
        "active_bad_angle_flags_is_excluded_by_instr_team": ([], True),
        "active_bad_angle_flags_is_suspected_transient": ([], False),
    }
    settings = PipelineSettings(xr.Dataset(data_vars))

    assert len(settings.active_bad_time_flags) == 17
    assert settings.active_bad_time_flags[6] is False  # is_night
    assert settings.active_bad_time_flags[16] is False  # is_spin_period_diff

    assert len(settings.active_bad_angle_flags) == 4
    assert settings.active_bad_angle_flags[3] is False  # is_suspected_transient
