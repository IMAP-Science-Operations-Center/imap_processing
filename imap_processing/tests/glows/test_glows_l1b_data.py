import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
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


@patch.object(
    HistogramL1B,
    "flag_uv_and_excluded",
    return_value=(np.zeros(3600, dtype=bool), np.zeros(3600, dtype=bool)),
)
@patch.object(HistogramL1B, "update_spice_parameters", autospec=True)
def test_validation_data_histogram(
    mock_spice_function,
    mock_flag_uv_and_excluded,
    l1a_dataset,
    mock_ancillary_exclusions,
    mock_pipeline_settings,
    mock_conversion_table_dict,
):
    mock_spice_function.side_effect = mock_update_spice_parameters
    ds = l1a_dataset[0]
    ds.attrs["flight_software_version"] = ds.attrs["flight_software_version"]
    ds.attrs["Parents"] = np.array(
        ["glows_test_packet_20110921_v01.pkts", "test_spice_file.tls"], dtype=object
    )

    # Only test with histogram data (l1a_dataset[0])
    l1b = glows_l1b(
        ds,
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
        epoch_val = met_to_ttj2000ns(
            validation_output["imap_start_time"]
            + validation_output["imap_end_time_offset"] / 2
        )

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


def test_get_threshold():
    "Test PipelineSettings.get_threshold method."

    test_data = {
        "n_sigma_threshold_lower": 3.0,
        "n_sigma_threshold_upper": 3.0,
        "relative_difference_threshold": 7e-05,
        "std_dev_threshold__celsius_deg": 2.03,
        "std_dev_threshold__volt": 50.0,
        "std_dev_threshold__sec": 0.033333,
        "std_dev_threshold__usec": 1.0,
    }
    pipeline_dataset = xr.Dataset({k: xr.DataArray(v) for k, v in test_data.items()})
    settings = PipelineSettings(pipeline_dataset)

    expected = [2.03, 50.0, 0.033333, 1.0, 7e-5]
    description = [
        "std_dev_threshold__celsius_deg",
        "std_dev_threshold__volt",
        "std_dev_threshold__sec",
        "std_dev_threshold__usec",
        "relative_difference_threshold",
    ]

    for name, exp in zip(description, expected, strict=False):
        threshold = settings.get_threshold(name)
        assert threshold == exp


@patch("imap_processing.glows.l1b.glows_l1b_data.geometry.imap_state")
@patch("imap_processing.glows.l1b.glows_l1b_data.get_instrument_spin_phase")
@patch("imap_processing.glows.l1b.glows_l1b_data.get_spin_data")
@patch("imap_processing.glows.l1b.glows_l1b_data.geometry.frame_transform")
@patch("imap_processing.glows.l1b.glows_l1b_data.sct_to_et")
@patch("imap_processing.glows.l1b.glows_l1b_data.met_to_sclkticks")
def test_update_spice_parameters_spin_axis_near_wrapping_point(
    mock_met_to_sclkticks,
    mock_sct_to_et,
    mock_frame_transform,
    mock_get_spin_data,
    mock_get_instrument_spin_phase,
    mock_imap_state,
):
    """Test spin axis orientation calculation near the longitude wrapping point.

    This test verifies that cartesian_to_latitudinal is called with degrees=False
    so that circmean/circstd receive values in radians. The bug was that without
    degrees=False, values in degrees would be passed to circmean/circstd which
    expect radians with low=-pi, high=pi.

    Test conditions:
    - Longitude values straddling +/-pi (wrapping point at 180 degrees)
    - Latitude near equator (-4 degrees)

    If the bug existed (degrees=True or default), circmean would receive values
    like 179 or -179 degrees when it expects radians in [-pi, pi]. This would
    produce nonsensical results because 179 >> pi.
    """
    # Mock time conversions - creates a time range of 5 seconds
    mock_met_to_sclkticks.return_value = 1000
    mock_sct_to_et.side_effect = lambda x: 100.0 if x == 1000 else 105.0

    # Mock spin data
    mock_spin_df = {
        "spin_start_met": np.array([99.0, 100.0, 101.0, 102.0, 103.0, 104.0]),
        "spin_period_sec": np.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0]),
    }
    mock_get_spin_data.return_value = pd.DataFrame(mock_spin_df)

    # Mock instrument spin phase
    mock_get_instrument_spin_phase.return_value = 0.5

    # Create cartesian vectors that straddle the longitude wrapping point.
    # Some points at +179 degrees and some at -179 degrees (which should
    # average to ~180 degrees when using proper circular mean).
    # Latitude near equator at -4 degrees.
    #
    # For spherical to cartesian (r=1):
    # x = cos(lat) * cos(lon)
    # y = cos(lat) * sin(lon)
    # z = sin(lat)
    lat_rad = np.deg2rad(-4.0)  # Near equator

    # Create 5 time steps: [+179, -179, +178, -178, +180] degrees longitude
    longitudes_deg = np.array([179.0, -179.0, 178.0, -178.0, 180.0])
    longitudes_rad = np.deg2rad(longitudes_deg)

    # Build cartesian vectors for each time step
    n_times = len(longitudes_rad)
    cartesian_vecs = np.zeros((n_times, 3))
    for i, lon_rad in enumerate(longitudes_rad):
        cartesian_vecs[i, 0] = np.cos(lat_rad) * np.cos(lon_rad)  # x
        cartesian_vecs[i, 1] = np.cos(lat_rad) * np.sin(lon_rad)  # y
        cartesian_vecs[i, 2] = np.sin(lat_rad)  # z

    mock_frame_transform.return_value = cartesian_vecs

    # Mock imap_state to return position and velocity for each time step
    mock_imap_state.return_value = np.tile(
        [[1e8, 2e8, 3e7, 10.0, 20.0, 5.0]], (n_times, 1)
    )

    # Create a minimal HistogramL1B-like object to test update_spice_parameters
    class MockHistogram:
        def __init__(self):
            self.imap_start_time = 100.0
            self.imap_time_offset = 5.0  # 5 second duration

    mock_hist = MockHistogram()

    # Call the actual update_spice_parameters method
    HistogramL1B.update_spice_parameters(mock_hist)

    # Verify spin_offset_correction shifts position_angle_offset_average by the
    # given amount.
    base_angle = mock_hist.position_angle_offset_average
    HistogramL1B.update_spice_parameters(mock_hist, spin_offset_correction=5.0)
    assert mock_hist.position_angle_offset_average == pytest.approx(base_angle + 5.0)

    # Verify the spin axis orientation values
    lon_result = mock_hist.spin_axis_orientation_average[0]
    lat_result = mock_hist.spin_axis_orientation_average[1]
    lon_std = mock_hist.spin_axis_orientation_std_dev[0]
    lat_std = mock_hist.spin_axis_orientation_std_dev[1]

    # The circular mean of [179, -179, 178, -178, 180] should be ~180 degrees
    # (or equivalently -180 degrees). The key test is that the result is NOT
    # near 0 degrees, which would happen if the values weren't properly handled
    # as circular data near the wrapping point.
    #
    # With the bug (degrees=True), circmean would receive [179, -179, ...] and
    # interpret these as radians, giving completely wrong results.
    #
    # Check that longitude is near 180 degrees (could be reported as -180)
    assert abs(abs(lon_result) - 180.0) < 5.0, (
        f"Longitude {lon_result} should be near +/-180 degrees. "
        "If near 0, the circular mean failed at the wrapping point."
    )

    # Latitude should be near -4 degrees
    np.testing.assert_allclose(lat_result, -4.0, atol=1.0)

    # Standard deviations should be small (all points are within a few degrees)
    assert lon_std < 5.0, f"Longitude std dev {lon_std} should be small"
    assert lat_std < 1.0, f"Latitude std dev {lat_std} should be small"
