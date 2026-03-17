import dataclasses
from unittest.mock import MagicMock

import numpy as np

from imap_processing.glows.l1a.glows_l1a import (
    create_glows_attr_obj,
    generate_de_dataset,
    generate_histogram_dataset,
)
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
