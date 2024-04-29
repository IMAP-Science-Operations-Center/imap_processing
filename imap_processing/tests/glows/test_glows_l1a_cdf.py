from pathlib import Path

import numpy as np
import pytest

from imap_processing.glows.l0 import decom_glows
from imap_processing.glows.l1.glows_l1a import (
    generate_de_dataset,
    generate_histogram_dataset,
    process_de_l0,
)
from imap_processing.glows.l1.glows_l1a_data import HistogramL1A


@pytest.fixture(scope="module")
def l1a_data():
    """Read test data from file"""
    current_directory = Path(__file__).parent
    packet_path = current_directory / "glows_test_packet_20110921_v01.pkts"
    histograms_l0, de_l0 = decom_glows.decom_packets(packet_path)

    histograms_l1a = [HistogramL1A(hist) for hist in histograms_l0]

    de_l1a_dict = process_de_l0(de_l0)

    de_l1a = []
    for _, value in de_l1a_dict.items():
        de_l1a += value

    return (histograms_l1a, de_l1a)


def test_generate_histogram_dataset(l1a_data):
    histograms_l1a, _ = l1a_data
    dataset = generate_histogram_dataset(histograms_l1a, "v001")

    assert (dataset["histograms"].data[0] == histograms_l1a[0].histograms).all()
    for key in histograms_l1a[0].block_header.keys():
        assert dataset[key].data[0] == histograms_l1a[0].block_header[key]
    assert (dataset["histograms"].data[-1] == histograms_l1a[-1].histograms).all()


def test_generate_de_dataset(l1a_data):
    _, de_l1a = l1a_data
    dataset = generate_de_dataset(de_l1a, "v001")
    assert len(dataset["epoch"].values) == len(de_l1a)

    # TODO fix this test
    assert (
        dataset["direct_events"].data[0] == np.pad(de_l1a[0].direct_events, 1389)
    ).all()
    for key in de_l1a[0].block_header.keys():
        assert dataset[key].data[0] == de_l1a[0].block_header[key]
    assert (dataset["direct_events"].data[-1] == de_l1a[-1].direct_events).all()


def test_glows_cdf_generation():
    # TODO: move this into a temporary directory

    expected_day_one = "20110920"
    expected_day_two = "20110921"

    # assert len(cdf_generation) == 2
    # assert expected_day_one in str(cdf_generation[0])
    # assert expected_day_two in str(cdf_generation[1])
