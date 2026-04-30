import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest

from imap_processing.glows.l0.decom_glows import decom_packets
from imap_processing.glows.l1a.glows_l1a import (
    create_glows_attr_obj,
    generate_de_dataset,
    generate_histogram_dataset,
    glows_l1a,
)
from imap_processing.glows.l1a.glows_l1a_data import HistogramL1A
from imap_processing.glows.utils.constants import TimeTuple


def test_generate_histogram_dataset(l1a_test_data):
    histogram_l1a, _ = l1a_test_data
    glows_attrs = create_glows_attr_obj()
    dataset = generate_histogram_dataset(histogram_l1a, glows_attrs)
    assert (dataset["histogram"].data[0] == histogram_l1a[0].histogram).all()
    hist_dict = dataclasses.asdict(histogram_l1a[0])
    for key, item in hist_dict.items():
        if key in [
            "imap_start_time",
            "imap_time_offset",
            "glows_start_time",
            "glows_time_offset",
        ]:
            assert (
                dataset[key].data[0]
                == TimeTuple(item["seconds"], item["subseconds"]).to_seconds()
            )
        elif key == "flags":
            assert dataset["flags_set_onboard"].data[0] == item["flags_set_onboard"]
            assert (
                dataset["is_generated_on_ground"].data[0]
                == item["is_generated_on_ground"]
            )

    for i in range(len(dataset["histogram"].data)):
        assert (dataset["histogram"].data[i] == histogram_l1a[i].histogram).all()


def test_generate_histogram_dataset_filters_empty(l1a_test_data):
    histogram_l1a, _ = l1a_test_data
    glows_attrs = create_glows_attr_obj()

    # Create an empty histogram (number_of_bins_per_histogram == 0)
    empty_hist = MagicMock()
    empty_hist.number_of_bins_per_histogram = 0
    empty_hist.histogram = []

    # Mix empty histograms into the list
    mixed_list = [empty_hist, histogram_l1a[0], empty_hist, histogram_l1a[1]]

    dataset = generate_histogram_dataset(mixed_list, glows_attrs)

    # Only the two non-empty histograms should appear in the output
    assert len(dataset["epoch"].values) == 2


def test_generate_histogram_dataset_filters_zero_imap_start_time(l1a_test_data):
    histogram_l1a, _ = l1a_test_data
    glows_attrs = create_glows_attr_obj()

    zero_time_hist = MagicMock()
    zero_time_hist.number_of_bins_per_histogram = 3600
    zero_time_hist.imap_start_time = TimeTuple(0, 0)

    mixed_list = [zero_time_hist, histogram_l1a[0], zero_time_hist, histogram_l1a[1]]

    dataset = generate_histogram_dataset(mixed_list, glows_attrs)

    assert len(dataset["epoch"].values) == 2


def test_generate_de_dataset(l1a_test_data):
    _, de_l1a = l1a_test_data
    glows_attrs = create_glows_attr_obj()

    dataset = generate_de_dataset(de_l1a, glows_attrs)
    non_none_len = len([de for de in de_l1a if de.de_data is not None])
    assert len(dataset["epoch"].values) == non_none_len

    # Output dataarrays are padded to the longest length in the entire set of packets.
    # Test data for the first and last DE need to be padded to this length
    assert (
        dataset["direct_events"].data[0]
        == np.pad(
            [event.to_list() for event in de_l1a[0].direct_events], ((0, 1389), (0, 0))
        )
    ).all()

    assert (
        dataset["direct_events"].data[-1]
        == np.pad(
            [event.to_list() for event in de_l1a[-1].direct_events], ((0, 651), (0, 0))
        )
    ).all()


@pytest.mark.external_test_data
def test_glows_l1a_no_zero_imap_start_time(in_flight_packet_path):
    hist_l0, _ = decom_packets(in_flight_packet_path)
    hist_l1a = [HistogramL1A(h) for h in hist_l0]
    non_empty = [h for h in hist_l1a if h.number_of_bins_per_histogram > 0]
    excluded = [h for h in non_empty if h.imap_start_time.seconds == 0]

    datasets = glows_l1a(in_flight_packet_path)
    hist_dataset = next(ds for ds in datasets if "hist" in ds.attrs["Logical_source"])
    assert (hist_dataset["imap_start_time"].values != 0).all()
    assert len(excluded) == 77
