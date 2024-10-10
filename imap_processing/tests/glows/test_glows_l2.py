import numpy as np

from imap_processing.glows.l2.glows_l2 import glows_l2, split_data_by_observational_day, \
    generate_l2


def test_glows_l2(l1b_hist_dataset):
    l2 = glows_l2(l1b_hist_dataset, "v001")


def test_split_by_observational_day(l1b_hist_dataset):
    split = split_data_by_observational_day(l1b_hist_dataset)
    l2 = generate_l2(split[0])


def test_generate_l2(l1b_hist_dataset):
    l2 = generate_l2(l1b_hist_dataset)

    expected_values = {
        "filter_temperature_average": [57.58],
        "filter_temperature_std_dev": [1.843e-01],
        "hv_voltage_average": [1715.4],
        "hv_voltage_std_dev": [2.274e-13]
    }

    assert np.isclose(l2.filter_temperature_average, expected_values["filter_temperature_average"], 0.01)
    assert np.isclose(l2.filter_temperature_variance, expected_values["filter_temperature_std_dev"], 0.01)
    assert np.isclose(l2.hv_voltage_average, expected_values["hv_voltage_average"], 0.01)
    assert np.isclose(l2.hv_voltage_variance, expected_values["hv_voltage_std_dev"], 0.01)