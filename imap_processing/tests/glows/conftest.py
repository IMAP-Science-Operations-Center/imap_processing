from functools import reduce
from pathlib import Path

import pytest

from imap_processing.glows.l0 import decom_glows
from imap_processing.glows.l1a.glows_l1a import glows_l1a, process_de_l0
from imap_processing.glows.l1a.glows_l1a_data import HistogramL1A
from imap_processing.glows.l1b.glows_l1b import glows_l1b


@pytest.fixture()
def packet_path():
    current_directory = Path(__file__).parent
    return current_directory / "validation_data" / "glows_test_packet_20110921_v01.pkts"


@pytest.fixture()
def decom_test_data(packet_path):
    """Read test data from file"""

    data_packet_list = decom_glows.decom_packets(packet_path)
    return data_packet_list


@pytest.fixture()
def l1a_test_data(decom_test_data):
    hist_l1a = []

    for hist in decom_test_data[0]:
        hist_l1a.append(HistogramL1A(hist))

    de_l1a_dict = process_de_l0(decom_test_data[1])

    # Flatten the dictionary to one list of DE values
    de_l1a = reduce(list.__add__, [value for value in de_l1a_dict.values()])

    return hist_l1a, de_l1a


@pytest.fixture()
def l1a_dataset(packet_path):
    return glows_l1a(packet_path, "v001")


@pytest.fixture()
def l1b_hist_dataset(l1a_dataset):
    return glows_l1b(l1a_dataset[0], "v001")
