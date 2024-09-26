import dataclasses
from functools import reduce
from pathlib import Path

import numpy as np
import pytest

from imap_processing.glows.l0 import decom_glows
from imap_processing.glows.l1a.glows_l1a import (
    create_glows_attr_obj,
    generate_de_dataset,
    generate_histogram_dataset,
    process_de_l0,
)
from imap_processing.glows.l1a.glows_l1a_data import HistogramL1A
from imap_processing.glows.utils.constants import TimeTuple


@pytest.fixture(scope="module")
def l1a_data():
    """Read test data from file"""
    current_directory = Path(__file__).parent
    packet_path = (
        current_directory / "validation_data" / "glows_test_packet_20110921_v01.pkts"
    )
    histogram_l0, de_l0 = decom_glows.decom_packets(packet_path)

    histogram_l1a = [HistogramL1A(hist) for hist in histogram_l0]

    de_l1a_dict = process_de_l0(de_l0)

    # Flatten the dictionary to one list of DE values
    de_l1a = reduce(list.__add__, [value for value in de_l1a_dict.values()])

    return (histogram_l1a, de_l1a)


def test_generate_histogram_dataset(l1a_data):
    histogram_l1a, _ = l1a_data
    glows_attrs = create_glows_attr_obj("v001")
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
        elif key not in ["histogram", "ground_software_version", "pkts_file_name"]:
            assert dataset[key].data[0] == item

    for i in range(len(dataset["histogram"].data)):
        assert (dataset["histogram"].data[i] == histogram_l1a[i].histogram).all()


def test_generate_de_dataset(l1a_data):
    _, de_l1a = l1a_data
    glows_attrs = create_glows_attr_obj("v001")
    dataset = generate_de_dataset(de_l1a, glows_attrs)
    assert len(dataset["epoch"].values) == len(de_l1a)

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
