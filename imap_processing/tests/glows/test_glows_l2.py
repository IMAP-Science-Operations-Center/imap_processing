import numpy as np
import xarray as xr

from imap_processing.glows.l2.glows_l2 import glows_l2, split_data_by_observational_day, \
    generate_l2, return_good_times


def test_glows_l2(l1b_hist_dataset):
    l2 = glows_l2(l1b_hist_dataset, "v001")


def test_split_by_observational_day(l1b_hist_dataset):
    split = split_data_by_observational_day(l1b_hist_dataset)
    l2 = generate_l2(split[0])

def test_filter_good_times():
    active_flags = np.ones((17,))
    active_flags[16] = 0
    test_flags = np.ones((4, 17))
    test_flags[1, 0] = 0
    test_flags[3, 16] = 0
    flags = xr.DataArray(test_flags, dims=["epoch", "flags"])

    good_times = return_good_times(flags, active_flags)
    expected_good_times = [0, 2, 3]

    assert np.array_equal(good_times, expected_good_times)


def test_generate_l2(l1b_hist_dataset):
    l2 = generate_l2(l1b_hist_dataset)

    expected_values = {
        "filter_temperature_average": [57.59],
        "filter_temperature_std_dev": [0.23],
        "hv_voltage_average": [1715.4],
        "hv_voltage_std_dev": [0.0]
    }

    assert np.isclose(l2.filter_temperature_average, expected_values["filter_temperature_average"], 0.01)
    assert np.isclose(l2.filter_temperature_std_dev, expected_values["filter_temperature_std_dev"], 0.01)
    assert np.isclose(l2.hv_voltage_average, expected_values["hv_voltage_average"], 0.01)
    assert np.isclose(l2.hv_voltage_std_dev, expected_values["hv_voltage_std_dev"], 0.01)