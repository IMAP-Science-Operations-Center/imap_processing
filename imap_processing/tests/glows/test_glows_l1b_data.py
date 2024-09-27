import json
from pathlib import Path

import numpy as np
import pytest

from imap_processing.cdf.utils import J2000_EPOCH
from imap_processing.glows.l1a.glows_l1a import glows_l1a
from imap_processing.glows.l1b.glows_l1b import glows_l1b
from imap_processing.glows.l1b.glows_l1b_data import AncillaryParameters, DirectEventL1B
from imap_processing.spice.time import met_to_j2000ns


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

def test_validation_data_histogram():
    input_data = Path(__file__).parent / "validation_data" / "glows_test_packet_20110921_v01.pkts"
    l1a_data = glows_l1a(input_data, "v001")
    hist_day_one = l1a_data[0]
    hist_day_two = l1a_data[1]

    l1b_day_one = glows_l1b(hist_day_one, "v001")
    l1b_day_two = glows_l1b(hist_day_two, "v001")

    validation_data = Path(__file__).parent / "validation_data" / "imap_glows_l1b_full_output.json"
    with open(validation_data, 'r') as f:
        out = json.load(f)

    # TODO block header, flags
    expected_matching_columns = [
                                 'glows_start_time',
                                 'glows_end_time_offset',
                                 'imap_start_time',
                                 'imap_end_time_offset',
                                 'number_of_spins_per_block',
                                 'number_of_bins_per_histogram',
                                 'histogram',
                                 'number_of_events',
                                 # 'imap_spin_angle_bin_cntr',
                                 # 'histogram_flag_array',
                                 'filter_temperature_average',
                                 'filter_temperature_std_dev',
                                 'hv_voltage_average',
                                 'hv_voltage_std_dev',
                                 'spin_period_average',
                                 'spin_period_std_dev',
                                 'pulse_length_average',
                                 'pulse_length_std_dev',
                                 # TODO uncomment when spice is complete
                                 # 'spin_period_ground_average',
                                 # 'spin_period_ground_std_dev',
                                 # 'position_angle_offset_average',
                                 # 'position_angle_offset_std_dev',
                                 # 'spin_axis_orientation_average',
                                 # 'spin_axis_orientation_std_dev',
                                 # 'spacecraft_location_average',
                                 # 'spacecraft_location_std_dev',
                                 # 'spacecraft_velocity_average',
                                 # 'spacecraft_velocity_std_dev',
    ]

    for index, validation_output in enumerate(out['output']):
        if validation_output['imap_start_time'] < 54259215:
            # day of 2011-09-20
            l1b = l1b_day_one
            l1b_index = index
        else:

            l1b_index = index - l1b_day_one.epoch.size
            l1b = l1b_day_two

        assert np.equal(validation_output['imap_start_time'],
                l1b.isel(epoch=l1b_index).imap_start_time.data)

        for key in validation_output:
            if key not in expected_matching_columns:
                continue

            np.testing.assert_array_almost_equal(l1b[key].isel(epoch=l1b_index).data, validation_output[key], decimal=1)

