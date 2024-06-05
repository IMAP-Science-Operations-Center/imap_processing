import pytest

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = f"{imap_module_directory}/tests/hit/PREFLIGHT_raw_record_2023_256_15_59_04_apid1251.pkts"  # noqa
    data_packet_list = hit_l1a.decom_packets(packet_file)
    return data_packet_list


def test_total_datasets(decom_test_data):
    """Test if total number of datasets is correct"""
    # assert len(decom_test_data) == total_datasets


def test_dataset_dims_length(decom_test_data):
    """Test if the time dimension length in the dataset is correct"""
    # assert decom_test_data["HIT_SCIENCE"].dims["epoch"] == num_packet_times
