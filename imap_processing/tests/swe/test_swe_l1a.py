import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.l0 import decom_swe
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
)
from imap_processing.utils import group_by_apid


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from test folder"""
    test_folder_path = "tests/swe/l0_data"
    packet_files = list(imap_module_directory.glob(f"{test_folder_path}/*.bin"))

    data_list = []
    for packet_file in packet_files:
        data_list.extend(decom_swe.decom_packets(packet_file))
    return data_list


def test_total_packets(decom_test_data):
    assert len(decom_test_data) == 23


def test_group_by_apid(decom_test_data):
    grouped_data = group_by_apid(decom_test_data)

    # check total dataset for swe science
    total_science_data = grouped_data[SWEAPID.SWE_SCIENCE]
    assert len(total_science_data) == 4

    # check total dataset for cem raw
    total_cem_raw_data = grouped_data[SWEAPID.SWE_CEM_RAW]
    assert len(total_cem_raw_data) == 2

    # check total dataset for housekeeping
    grouped_data[SWEAPID.SWE_APP_HK]
    assert len(total_cem_raw_data) == 2

    # check total dataset for event message
    total_event_message_data = grouped_data[SWEAPID.SWE_EVTMSG]
    assert len(total_event_message_data) == 15


def test_cdf_creation(decom_test_data):
    grouped_data = group_by_apid(decom_test_data)
    sci_cdf_filepath = swe_l1a(grouped_data[SWEAPID.SWE_SCIENCE])
    hk_cdf_filepath = swe_l1a(grouped_data[SWEAPID.SWE_APP_HK])
    assert sci_cdf_filepath.name == "imap_swe_l1a_sci_20230927_v01.cdf"
    assert hk_cdf_filepath.name == "imap_swe_l1a_lveng-hk_20230927_v01.cdf"
