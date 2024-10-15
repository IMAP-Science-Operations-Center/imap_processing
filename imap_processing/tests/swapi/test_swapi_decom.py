import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets
from imap_processing.swapi.l1.swapi_l1 import (
    SWAPIAPID,
)
from imap_processing.utils import group_by_apid


@pytest.fixture(scope="session")
def decom_test_data(swapi_l0_test_data_path):
    """Read test data from file"""
    test_file = "imap_swapi_l0_raw_20240924_v001.pkts"
    packet_file = imap_module_directory / swapi_l0_test_data_path / test_file
    packet_definition = (
        f"{imap_module_directory}/swapi/packet_definitions/swapi_packet_definition.xml"
    )
    data_list = []
    data_list.extend(decom_packets(packet_file, packet_definition))
    return data_list


def test_number_of_packets(decom_test_data):
    """This test and validate number of packets."""
    grouped_data = group_by_apid(decom_test_data)
    sci_packets = grouped_data[SWAPIAPID.SWP_SCI]
    expected_sci_packets = 153
    assert len(sci_packets) == expected_sci_packets

    hk_packets = grouped_data[SWAPIAPID.SWP_HK]
    expected_hk_packets = 17
    assert len(hk_packets) == expected_hk_packets


def test_swapi_sci_data(decom_test_data, swapi_l0_validation_data_path):
    """This test and validate raw data of SWAPI raw science data."""
    # read validation data
    raw_validation_data = pd.read_csv(
        swapi_l0_validation_data_path / "idle_export_raw.SWP_SCI_20240924_080204.csv",
        index_col="SHCOARSE",
    )

    grouped_data = group_by_apid(decom_test_data)
    sci_packets = grouped_data[SWAPIAPID.SWP_SCI]
    first_data = sci_packets[0]
    validation_data = raw_validation_data.loc[first_data["SHCOARSE"]]

    # compare raw values of validation data
    for key, value in first_data.items():
        # check if the data is the same
        if key == "PLAN_ID_SCIENCE":
            # We had to work around this because HK and SCI packet uses
            # PLAN_ID but they uses different length of bits.
            assert value == validation_data["PLAN_ID"]
        elif key == "SPARE_2_SCIENCE":
            # Same for this SPARE_2 as above case
            assert value == validation_data["SPARE_2"]
        elif key == "MODE":
            assert value.raw_value == validation_data[key]
        elif "RNG" in key:
            assert value.raw_value == validation_data[key]
        else:
            # for SHCOARSE we need the name of the column.
            # This is done because pandas removed it from the
            # main columns to make it the index.
            assert value.raw_value == (
                validation_data[key] if key != "SHCOARSE" else validation_data.name
            )


def test_swapi_hk_data(decom_test_data, swapi_l0_validation_data_path):
    """This test and validate raw data of SWAPI raw housekeeping data."""
    # read validation data
    raw_validation_data = pd.read_csv(
        swapi_l0_validation_data_path / "idle_export_raw.SWP_HK_20240924_080204.csv",
        index_col="SHCOARSE",
    )

    grouped_data = group_by_apid(decom_test_data)
    hk_packets = grouped_data[SWAPIAPID.SWP_HK]
    first_data = hk_packets[0]
    validation_data = raw_validation_data.loc[first_data["SHCOARSE"]]
    bad_keys = [
        "N5_V",
        "SCEM_I",
        "P5_I",
        "PHD_LLD1_V",
        "SPARE_4",
        "P_CEM_CMD_LVL_MON",
        "S_CEM_CMD_LVL_MON",
        "ESA_CMD_LVL_MON",
        "PHD_LLD2_V",
        "CHKSUM",
    ]
    # compare raw values of validation data
    for key, value in first_data.items():
        if key == "PLAN_ID_HK":
            # We had to work around this because HK and SCI packet uses
            # PLAN_ID but they uses different length of bits.
            assert value == validation_data["PLAN_ID"]
        elif key == "SPARE_2_HK":
            # Same for this SPARE_2 as PLAN_ID
            assert value == validation_data["SPARE_2"]
        elif key == "SHCOARSE":
            # for SHCOARSE we need the name of the column.
            # This is done because pandas removed it from the main columns
            # to make it the index.
            assert value == validation_data.name
        elif key in bad_keys:
            # TODO: remove this elif after getting good validation data
            # Validation data has wrong value for N5_V
            continue
        else:
            assert value.raw_value == validation_data[key]
