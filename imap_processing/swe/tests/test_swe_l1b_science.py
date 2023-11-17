import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.l0 import decom_swe
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b_science import (
    get_indices_of_full_cycles,
)


@pytest.fixture(scope="session")
def l1a_test_data():
    """Read test data from file"""
    # NOTE: data was provided in this sequence in both bin and validation data
    # from instrument team.
    # Packet 1 has spin 4's data
    # Packet 2 has spin 1's data
    # Packet 3 has spin 2's data
    # Packet 4 has spin 3's data
    # moved packet 1 to bottom to show data in order.
    packet_files = [
        f"{imap_module_directory}/swe/tests/l0_data/20230927173253_SWE_SCIENCE_packet.bin",
        f"{imap_module_directory}/swe/tests/l0_data/20230927173308_SWE_SCIENCE_packet.bin",
        f"{imap_module_directory}/swe/tests/l0_data/20230927173323_SWE_SCIENCE_packet.bin",
        f"{imap_module_directory}/swe/tests/l0_data/20230927173238_SWE_SCIENCE_packet.bin",
    ]
    data = []
    for packet_file in packet_files:
        data.extend(decom_swe.decom_packets(packet_file))
    # Get unpacked science data
    unpacked_data = swe_science(data)
    return unpacked_data


def test_get_full_cycle_data_indices():
    q = np.array([0, 1, 2, 0, 1, 2, 3, 2, 3, 0, 2, 3, 0, 1, 2, 3, 2, 3, 1, 0])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([3, 4, 5, 6, 12, 13, 14, 15]))

    q = np.array([0, 1, 0, 1, 2, 3, 0, 2])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([2, 3, 4, 5]))

    q = np.array([0, 1, 2, 3])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([0, 1, 2, 3]))

    q = np.array([1, 2])
    filtered_q = get_indices_of_full_cycles(q)
    np.testing.assert_array_equal(filtered_q, np.array([]))
