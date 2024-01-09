import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.l0 import decom_swe


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_1_file = (
        imap_module_directory / "tests/swe/l0_data/20230927100248_SWE_HK_packet.bin"
    )
    packet_2_file = (
        imap_module_directory / "tests/swe/l0_data/20230927100412_SWE_HK_packet.bin"
    )
    first_data = decom_swe.decom_packets(packet_1_file)
    second_data = decom_swe.decom_packets(packet_2_file)

    return first_data + second_data


def test_number_of_packets(decom_test_data):
    """This test and validate number of packets."""
    expected_number_of_packets = 2
    assert len(decom_test_data) == expected_number_of_packets


def test_swe_raw_housekeeping_data(decom_test_data):
    """This test and validate raw and derived data of SWE housekeeping data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    raw_validation_data = pd.read_csv(
        test_data_path / "data_raw.SWE_APP_HK_20230927_094839.csv",
        index_col="SHCOARSE",
    )

    first_data = decom_test_data[0]
    validation_data = raw_validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # compare raw values of housekeeping data
    for key, value in first_data.data.items():
        if key == "SHCOARSE":
            # compare SHCOARSE value
            assert value.raw_value == validation_data.name
            continue
        # check if the data is the same
        assert value.raw_value == validation_data[key]


def test_swe_derived_housekeeping_data(decom_test_data):
    """This test and validate derived data of SWE housekeeping data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    derived_validation_data = pd.read_csv(
        test_data_path / "data_derived.SWE_APP_HK_20230927_094839.csv",
        index_col="SHCOARSE",
    )
    second_data = decom_test_data[1]
    validation_data = derived_validation_data.loc[
        second_data.data["SHCOARSE"].raw_value
    ]
    enum_name_list = [
        "APP_MODE",
        "SAFED",
        "LAST_ACC_OPCODE",
        "HV_DISABLE_PLUG",
        "HV_LIMIT_PLUG",
        "HVPS_ENABLE",
        "HVPS_CEM_ENABLE",
        "HVPS_ESA_ENABLE",
        "HVPS_BULK_ENABLE",
        "SENSOR_P12A_N12A_CTRL",
        "SENSOR_P5A_N5A_CTRL",
        "SENSOR_P3P3D_P5D_CTRL",
        "STIM_ENABLE",
        "FDC_LAST_TRIGGER_ACTION",
        "ACTIVE_MACRO_TRIGGER",
        "FDC_LAST_TRIGGER_MINMAX",
    ]

    # Check ENUM values from housekeeping data
    for enum_name in enum_name_list:
        assert second_data.data[enum_name].derived_value == validation_data[enum_name]
