from dataclasses import fields

import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.swe.utils.swe_utils import SWEAPID
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def decom_raw_cem_data():
    """Read test data from file"""
    test_folder_path = imap_module_directory / "tests/swe/l0_data"
    packet_file = test_folder_path / "2024051011_SWE_CEM_RAW_packet.bin"
    xtce_document = (
        imap_module_directory / "swe/packet_definitions/swe_packet_definition.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file, xtce_document, use_derived_value=False
    )
    return datasets_by_apid[SWEAPID.SWE_CEM_RAW]


def test_number_of_packets(decom_raw_cem_data):
    """This test and validate number of packets."""
    expected_number_of_packets = 25
    assert len(decom_raw_cem_data) == expected_number_of_packets


def test_swe_raw_cem_data(decom_raw_cem_data):
    """This test and validate raw data of SWE raw CEM data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    raw_validation_data = pd.read_csv(
        test_data_path / "idle_export_eu.SWE_CEM_RAW_20240510_092742.csv",
        index_col="SHCOARSE",
    )

    first_data = decom_raw_cem_data.isel(epoch=0)
    validation_data = raw_validation_data.loc[first_data["shcoarse"].values]

    ccsds_header_keys = [field.name for field in fields(CcsdsData)]
    ccsds_header_keys += ["SHCOARSE"]

    # compare unpacked data to validation data
    for key in first_data.keys():
        if key.upper() in ccsds_header_keys:
            continue
        # check if the data is the same.
        assert first_data[key] == validation_data[key.upper()]
