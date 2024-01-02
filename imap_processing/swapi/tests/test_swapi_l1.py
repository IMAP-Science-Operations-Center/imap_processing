import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets
from imap_processing.swapi.l1.swapi_l1 import SWAPIAPID, process_swapi_science
from imap_processing.swapi.swapi_utils import create_dataset
from imap_processing.utils import group_by_apid, sort_by_time


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    test_folder_path = "swapi/tests/l0_data"
    packet_files = list(imap_module_directory.glob(f"{test_folder_path}/*.bin"))
    packet_definition = (
        f"{imap_module_directory}/swapi/packet_definitions/swapi_packet_definition.xml"
    )
    data_list = []
    for packet_file in packet_files:
        data_list.extend(decom_packets(packet_file, packet_definition))
    return data_list


def test_decom_swapi_algorithm(decom_test_data):
    grouped_data = group_by_apid(decom_test_data)
    science_data = grouped_data[SWAPIAPID.SWP_SCI]
    sorted_packets = sort_by_time(science_data, "SHCOARSE")
    ds_data = create_dataset(sorted_packets)
    process_swapi_science(ds_data)
