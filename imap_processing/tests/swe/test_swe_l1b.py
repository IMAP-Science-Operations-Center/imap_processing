import pandas as pd
import pytest
from cdflib.xarray import cdf_to_xarray

from imap_processing import imap_module_directory
from imap_processing.swe.l0 import decom_swe
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
    create_dataset,
)
from imap_processing.utils import convert_raw_to_eu, group_by_apid


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from test folder"""
    test_folder_path = "tests/swe/l0_data"
    packet_files = list(imap_module_directory.glob(f"{test_folder_path}/*.bin"))

    data_list = []
    for packet_file in packet_files:
        data_list.extend(decom_swe.decom_packets(packet_file))
    return data_list


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
        imap_module_directory
        / "tests/swe/l0_data/20230927173253_SWE_SCIENCE_packet.bin",
        imap_module_directory
        / "tests/swe/l0_data/20230927173308_SWE_SCIENCE_packet.bin",
        imap_module_directory
        / "tests/swe/l0_data/20230927173323_SWE_SCIENCE_packet.bin",
        imap_module_directory
        / "tests/swe/l0_data/20230927173238_SWE_SCIENCE_packet.bin",
    ]
    data = []
    for packet_file in packet_files:
        data.extend(decom_swe.decom_packets(packet_file))
    # Get unpacked science data
    unpacked_data = swe_science(data)
    return unpacked_data


def test_swe_l1b(decom_test_data):
    """Test that calculate engineering unit(EU) is
    matches validation data.

    Parameters
    ----------
    decom_test_data : list
        List of packets
    """
    # SWE_APP_HK = 1330
    # SWE_SCIENCE = 1344
    grouped_data = group_by_apid(decom_test_data)
    # Process science to l1a.
    # because of test data being in the wrong
    # order, we need to manually re-sort data
    # into order.
    sorted_packets = sorted(
        grouped_data[1344], key=lambda x: x.data["QUARTER_CYCLE"].raw_value
    )
    science_l1a_ds = swe_science(sorted_packets)
    # convert value from raw to engineering units as needed
    conversion_table_path = (
        imap_module_directory / "swe/l1b/engineering_unit_convert_table.csv"
    )
    # Look up packet name from APID
    packet_name = SWEAPID.SWE_SCIENCE.name
    # Convert raw data to engineering units as needed
    science_l1b = convert_raw_to_eu(
        science_l1a_ds,
        conversion_table_path=conversion_table_path,
        packet_name=packet_name,
    )

    # read science validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    eu_validation_data = pd.read_csv(
        test_data_path / "idle_export_eu.SWE_SCIENCE_20230927_172708.csv",
        index_col="SHCOARSE",
    )
    second_data = sorted_packets[1]
    validation_data = eu_validation_data.loc[second_data.data["SHCOARSE"].raw_value]

    science_eu_field_list = [
        "SPIN_PHASE",
        "SPIN_PERIOD",
        "THRESHOLD_DAC",
    ]

    # Test EU values for science data
    for field in science_eu_field_list:
        assert round(science_l1b[field].data[1], 5) == round(validation_data[field], 5)

    # process housekeeping data to l1a and create l1b
    hk_l1a_ds = create_dataset(grouped_data[1330])
    hk_l1b = convert_raw_to_eu(
        hk_l1a_ds, conversion_table_path, SWEAPID.SWE_APP_HK.name
    )

    # read housekeeping validation data
    eu_validation_data = pd.read_csv(
        test_data_path / "data_derived.SWE_APP_HK_20230927_094839.csv",
        index_col="SHCOARSE",
    )
    second_data = grouped_data[1330][1]
    validation_data = eu_validation_data.loc[second_data.data["SHCOARSE"].raw_value]

    # check that these field's calculated EU value matches with
    # validation data's EU value.
    eu_field_list = [
        "HVPS_VBULK",
        "HVPS_VCEM",
        "HVPS_VESA",
        "HVPS_VESA_LOW_RANGE",
        "HVPS_ICEM",
        "FEE_TEMP",
        "SENSOR_TEMP",
        "HVPS_TEMP",
        "LVPS_BACK_BOARD_TEMP",
    ]

    # Check EU values from housekeeping data
    for field in eu_field_list:
        assert round(hk_l1b[field].data[1], 5) == round(validation_data[field], 5)


def test_cdf_creation(decom_test_data, l1a_test_data):
    sci_l1b_filepath = swe_l1b(l1a_test_data)

    # process hk data to l1a and then pass to l1b
    grouped_data = group_by_apid(decom_test_data)
    # writes data to CDF file
    hk_l1a_filepath = swe_l1a(grouped_data[SWEAPID.SWE_APP_HK])
    # reads data from CDF file and passes to l1b
    l1a_dataset = cdf_to_xarray(hk_l1a_filepath)
    hk_l1b_filepath = swe_l1b(l1a_dataset)

    assert hk_l1b_filepath.name == "imap_swe_l1b_lveng-hk_20230927_v01.cdf"
    assert sci_l1b_filepath.name == "imap_swe_l1b_sci_20230927_v01.cdf"
