import pytest

from imap_processing.hit.l0 import hit_l1a_decom


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = (
        "imap_processing/hit/tests/PREFLIGHT_raw_record_2023_256_15_59_04_apid1252.pkts"
    )
    xtce = "imap_processing/hit/packet_definitions/P_HIT_SCIENCE.xml"
    data_packet_list = hit_l1a_decom.decom_hit_packets(packet_file, xtce)
    return data_packet_list


def test_total_datasets(decom_test_data):
    """Test if total number of datasets is correct"""
    total_datasets = 1
    assert len(decom_test_data) == total_datasets


def test_dataset_dims_length(decom_test_data):
    """Test if the time dimension length in the dataset is correct"""
    num_packet_times = 1130
    assert decom_test_data["HIT_SCIENCE"].dims["Epoch"] == num_packet_times
