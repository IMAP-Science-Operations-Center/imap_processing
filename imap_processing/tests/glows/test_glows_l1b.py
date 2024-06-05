import dataclasses

import numpy as np
import pytest
import xarray as xr

from imap_processing.glows.l1b.glows_l1b import histogram_mapping, process_histogram
from imap_processing.glows.l1b.glows_l1b_data import HistogramL1B


@pytest.fixture()
def hist_dataset():
    variables = {
        "flight_software_version": np.zeros((20,)),
        "seq_count_in_pkts_file": np.zeros((20,)),
        "last_spin_id": np.zeros((20,)),
        "flags_set_onboard": np.zeros((20,)),
        "is_generated_on_ground": np.zeros((20,)),
        "number_of_spins_per_block": np.zeros((20,)),
        "number_of_bins_per_histogram": np.zeros((20,)),
        "number_of_events": np.zeros((20,)),
        "filter_temperature_average": np.zeros((20,)),
        "filter_temperature_variance": np.zeros((20,)),
        "hv_voltage_average": np.zeros((20,)),
        "hv_voltage_variance": np.zeros((20,)),
        "spin_period_average": np.zeros((20,)),
        "spin_period_variance": np.zeros((20,)),
        "pulse_length_average": np.zeros((20,)),
        "pulse_length_variance": np.zeros((20,)),
        "imap_start_time": np.zeros((20,)),
        "imap_time_offset": np.zeros((20,)),
        "glows_start_time": np.zeros((20,)),
        "glows_time_offset": np.zeros((20,)),
    }
    epoch = xr.DataArray(np.arange(20), name="epoch", dims=["epoch"])

    bins = xr.DataArray(np.arange(3600), name="bins", dims=["bins"])

    ds = xr.Dataset(coords={"epoch": epoch})
    ds["histograms"] = xr.DataArray(
        np.zeros((20, 3600)),
        dims=["epoch", "bins"],
        coords={"epoch": epoch, "bins": bins},
    )

    for var in variables:
        ds[var] = xr.DataArray(variables[var], dims=["epoch"], coords={"epoch": epoch})

    return ds


def test_histogram_mapping():
    time_val = 1111111.11
    # A = 2.318
    # B = 69.5454
    expected_temp = 100

    test_hists = np.zeros((200, 3600))
    # For temp
    encoded_val = expected_temp * 2.318 + 69.5454

    # For now, testing types and number of inputs
    output = histogram_mapping(
        test_hists,
        "test",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        time_val,
        time_val,
        time_val,
        time_val,
    )

    assert output[17] == time_val

    # Correctly decoded temperature
    assert output[9] - expected_temp < 0.1


def test_process_histograms(hist_dataset):
    time_val = np.single(1111111.11)
    # A = 2.318
    # B = 69.5454
    expected_temp = 100

    test_hists = np.zeros((200,))
    # For temp
    encoded_val = np.single(expected_temp * 2.318 + 69.5454)

    test_l1b = HistogramL1B(
        test_hists,
        "test",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        encoded_val,
        time_val,
        time_val,
        time_val,
        time_val,
    )

    output = process_histogram(hist_dataset)

    assert len(output) == len(dataclasses.asdict(test_l1b).keys())
