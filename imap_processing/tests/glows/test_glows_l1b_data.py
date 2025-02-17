import json
from pathlib import Path

import numpy as np
import pytest

from imap_processing.glows.l1b.glows_l1b import glows_l1b
from imap_processing.glows.l1b.glows_l1b_data import (
    AncillaryParameters,
    DirectEventL1B,
    HistogramL1B,
)
from imap_processing.spice.time import met_to_ttj2000ns


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
    for key in fake_good_input:
        assert getattr(ancillary, key) == fake_good_input[key]

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


def test_validation_data_histogram(l1a_dataset):
    l1b = [glows_l1b(l1a_dataset[0], "v001"), glows_l1b(l1a_dataset[1], "v001")]
    end_time = l1b[0]["epoch"].data[-1]

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
        "number_of_spins_per_block": "number_of_spins_per_block",
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

        # Validation data spans the two obs days, so this selects the correct output
        dataset_index = 1 if epoch_val > end_time else 0
        datapoint = l1b[dataset_index].sel(epoch=epoch_val)

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


def test_validation_data_de(l1a_dataset):
    de_data = l1a_dataset[2]

    l1b = glows_l1b(de_data, "v001")
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
