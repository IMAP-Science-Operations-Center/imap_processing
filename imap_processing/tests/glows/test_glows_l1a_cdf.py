from pathlib import Path

import pytest

from imap_processing.glows.l0 import decom_glows
from imap_processing.glows.l1.glows_l1a import (
    generate_histogram_dataset,
    glows_l1a,
)
from imap_processing.glows.l1.glows_l1a_data import DirectEventL1A, HistogramL1A


@pytest.fixture(scope="module")
def l1a_data():
    """Read test data from file"""
    current_directory = Path(__file__).parent
    packet_path = current_directory / "glows_test_packet_20110921_v01.pkts"
    histograms_l0, de_l0 = decom_glows.decom_packets(packet_path)

    histograms_l1a = [HistogramL1A(hist) for hist in histograms_l0]
    de_l1a = [DirectEventL1A(de) for de in de_l0]
    return (histograms_l1a, de_l1a)


def test_generate_histogram_dataset(l1a_data):
    histograms_l1a, _ = l1a_data
    dataset = generate_histogram_dataset(histograms_l1a, "v001")

    assert (dataset["histograms"].data[0] == histograms_l1a[0].histograms).all()
    for key in histograms_l1a[0].block_header.keys():
        assert dataset[key].data[0] == histograms_l1a[0].block_header[key]
    assert (dataset["histograms"].data[-1] == histograms_l1a[-1].histograms).all()


def test_glows_cdf_generation():
    current_directory = Path(__file__).parent
    packet_path = current_directory / "glows_test_packet_20110921_v01.pkts"
    output_files = glows_l1a(packet_path, "v001")
    expected_day_one = "2011-09-20"
    expected_day_two = "2011-09-21"

    print(output_files)

    assert len(output_files) == 2
    assert expected_day_one in str(output_files[0])
    assert expected_day_two in str(output_files[1])
