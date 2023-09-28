import pytest

from imap_processing.swe.l0 import decom_swe
from imap_processing.swe.l1a.swe_science import swe_science, uncompress_counts


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = "imap_processing/swe/tests/science_block_20221116_163611Z_idle.bin"
    data_packet_list = decom_swe.decom_packets(packet_file)
    return data_packet_list


def test_uncompress_algorithm():
    """Test that we get correct uncompressed counts from the algorithm."""
    expected_value = 24063
    input_count = 230
    returned_value = uncompress_counts(input_count)
    assert expected_value == returned_value


def test_swe_science_algorithm(decom_test_data):
    """Test general shape of return dataset from swe_science.
    TODO: test expected values when we get validation data from
    SWE team in couple of weeks.
    """
    # First ESA in test data is 1.
    data = swe_science(decom_test_data)
    first_esa_steps = data["ESA_STEPS"][0]
    assert first_esa_steps == 1

    # science data should have this shape, 15x12x7.
    science_data = data["SCIENCE_DATA"][0]
    assert science_data.shape == (15, 12, 7)

    # Test data has 23 packets, therefore, SPIN_PHASE should have this length.
    spin_phase = data["SPIN_PHASE"]
    assert len(spin_phase) == 23
