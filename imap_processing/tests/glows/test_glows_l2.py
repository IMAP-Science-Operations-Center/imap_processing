import numpy as np
import pytest
import xarray as xr

from imap_processing.glows.l2.glows_l2 import (
    generate_l2,
    glows_l2,
    return_good_times,
)
from imap_processing.glows.l2.glows_l2_data import DailyLightcurve


@pytest.fixture()
def l1b_hists():
    epoch = xr.DataArray(np.arange(4), name="epoch", dims=["epoch"])
    bins = xr.DataArray(np.arange(5), name="bins", dims=["bins"])
    hist = xr.DataArray(
        np.ones((4, 5)), dims=["epoch", "bins"], coords={"epoch": epoch, "bins": bins}
    )
    hist[1, 0] = -1
    hist[2, 0] = -1
    hist[1, 1] = -1
    hist[2, 3] = -1

    input = xr.Dataset(coords={"epoch": epoch, "bins": bins})
    input["histogram"] = hist

    return input


def test_glows_l2(l1b_hist_dataset):
    l2 = glows_l2(l1b_hist_dataset, "v001")[0]
    assert l2.attrs["Logical_source"] == "imap_glows_l2_hist"

    assert np.allclose(l2["filter_temperature_average"].values, [57.6], rtol=0.1)


@pytest.mark.skip(reason="Spin table not yet complete")
def test_split_by_observational_day(l1b_hist_dataset):
    # TODO: Complete test when spin table is complete
    raise NotImplementedError


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
        "filter_temperature_std_dev": [0.21],
        "hv_voltage_average": [1715.4],
        "hv_voltage_std_dev": [0.0],
    }

    assert np.isclose(
        l2.filter_temperature_average,
        expected_values["filter_temperature_average"],
        0.01,
    )
    assert np.isclose(
        l2.filter_temperature_std_dev,
        expected_values["filter_temperature_std_dev"],
        0.01,
    )
    assert np.isclose(
        l2.hv_voltage_average, expected_values["hv_voltage_average"], 0.01
    )
    assert np.isclose(
        l2.hv_voltage_std_dev, expected_values["hv_voltage_std_dev"], 0.01
    )


def test_exposure_times(l1b_hists):
    exposure_time = xr.DataArray([10, 10, 20, 10])
    expected_times = np.array([20, 40, 50, 30, 50])

    times = DailyLightcurve.calculate_exposure_times(l1b_hists, exposure_time)

    assert np.array_equal(times, expected_times)


def test_bin_exclusions(l1b_hists):
    # TODO test excluding bins as well

    raw_hists = DailyLightcurve.calculate_histogram_sums(l1b_hists["histogram"].data)
    expected_values = [2, 3, 4, 3, 4]

    assert np.array_equal(raw_hists, expected_values)
