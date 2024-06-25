import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets
from imap_processing.swapi.l1.swapi_l1 import (
    SWAPIAPID,
)
from imap_processing.utils import group_by_apid


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    test_folder_path = "tests/swapi/l0_data"
    packet_files = list(imap_module_directory.glob(f"{test_folder_path}/*.pkts"))
    packet_definition = (
        f"{imap_module_directory}/swapi/packet_definitions/swapi_packet_definition.xml"
    )
    data_list = []
    for packet_file in packet_files:
        data_list.extend(decom_packets(packet_file, packet_definition))
    return data_list


def test_number_of_packets(decom_test_data):
    """This test and validate number of packets."""
    grouped_data = group_by_apid(decom_test_data)
    sci_packets = grouped_data[SWAPIAPID.SWP_SCI]
    expected_sci_packets = 54
    assert len(sci_packets) == expected_sci_packets

    hk_packets = grouped_data[SWAPIAPID.SWP_HK]
    expected_hk_packets = 54
    assert len(hk_packets) == expected_hk_packets

    aut_packets = grouped_data[SWAPIAPID.SWP_AUT]
    expected_aut_packets = 54
    assert len(aut_packets) == expected_aut_packets


def test_swapi_sci_data(decom_test_data):
    """This test and validate raw data of SWAPI raw science data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swapi/l0_validation_data"
    raw_validation_data = pd.read_csv(
        test_data_path / "idle_export_eu.SWP_SCI_20231012_125245.csv",
        index_col="SHCOARSE",
    )

    grouped_data = group_by_apid(decom_test_data)
    sci_packets = grouped_data[SWAPIAPID.SWP_SCI]
    first_data = sci_packets[0]
    validation_data = raw_validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # compare raw values of validation data
    for key, value in first_data.data.items():
        # check if the data is the same
        if key == "PLAN_ID_SCIENCE":
            # We had to work around this because HK and SCI packet uses
            # PLAN_ID but they uses different length of bits.
            assert value.raw_value == validation_data["PLAN_ID"]
        elif key == "SPARE_2_SCIENCE":
            # Same for this SPARE_2 as above case
            assert value.raw_value == validation_data["SPARE_2"]
        elif key == "MODE":
            # Because validation data uses derived value instead of raw value
            assert value.derived_value == validation_data[key]
        elif "RNG" in key:
            assert value.derived_value == validation_data[key]
        else:
            # for SHCOARSE we need the name of the column.
            # This is done because pandas removed it from the
            # main columns to make it the index.
            assert value.raw_value == (
                validation_data[key] if key != "SHCOARSE" else validation_data.name
            )


def test_swapi_hk_data(decom_test_data):
    """This test and validate raw data of SWAPI raw housekeeping data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swapi/l0_validation_data"
    raw_validation_data = pd.read_csv(
        test_data_path / "idle_export_raw.SWP_HK_20231012_125245.csv",
        index_col="SHCOARSE",
    )

    grouped_data = group_by_apid(decom_test_data)
    hk_packets = grouped_data[SWAPIAPID.SWP_HK]
    first_data = hk_packets[0]
    validation_data = raw_validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # compare raw values of validation data
    for key, value in first_data.data.items():
        if key == "PLAN_ID_HK":
            # We had to work around this because HK and SCI packet uses
            # PLAN_ID but they uses different length of bits.
            assert value.raw_value == validation_data["PLAN_ID"]
        elif key == "SPARE_2_HK":
            # Same for this SPARE_2 as PLAN_ID
            assert value.raw_value == validation_data["SPARE_2"]
        elif key == "SHCOARSE":
            # for SHCOARSE we need the name of the column.
            # This is done because pandas removed it from the main columns
            # to make it the index.
            assert value.raw_value == validation_data.name
        elif key == "N5_V":
            # TODO: remove this elif after getting good validation data
            # Validation data has wrong value for N5_V
            continue
        else:
            assert value.raw_value == validation_data[key]


def test_swapi_aut_data(decom_test_data):
    """This test and validate raw data of SWAPI raw autonomy data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swapi/l0_validation_data"
    raw_validation_data = pd.read_csv(
        test_data_path / "idle_export_raw.SWP_AUT_20231012_125245.csv",
        index_col="SHCOARSE",
    )

    grouped_data = group_by_apid(decom_test_data)
    aut_packets = grouped_data[SWAPIAPID.SWP_AUT]
    first_data = aut_packets[0]
    validation_data = raw_validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # compare raw values of science data
    for key, value in first_data.data.items():
        if key == "SHCOARSE":
            assert value.raw_value == validation_data.name
        elif key == "SPARE_1_AUT":
            # We had to work around this because HK and SCI packet uses
            # SPARE_1 but they uses different length of bits.
            assert value.raw_value == validation_data["SPARE_1"]
        else:
            assert value.raw_value == validation_data[key]
