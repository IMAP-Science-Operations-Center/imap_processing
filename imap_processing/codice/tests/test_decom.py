"""Tests the decommutation process for CoDICE CCSDS Packets. This also tests the
unit conversion process for CoDICE housekeeping data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.codice.l0 import decom_codice


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = Path(
        f"{imap_module_directory}/codice/tests/data/"
        f"raw_ccsds_20230822_122700Z_idle.bin"
    )
    Path(f"{imap_module_directory}/codice/packet_definitions/P_COD_NHK.xml")
    data_packet_list = decom_codice.decom_packets(packet_file)
    data_packet_list = [
        packet
        for packet in data_packet_list
        if packet.header["PKT_APID"].raw_value == 1136
    ]

    return data_packet_list


@pytest.fixture(scope="session")
def validation_data():
    """Read in validation data from the CSV file

    Returns
    -------
    validation_data : pandas DataFrame
        The validation data read from the CSV, cleaned up and ready to compare
        the decommutated packet with
    """

    # Read in the CSV file (perhaps to a pandas dataframe?)
    validation_file = Path(
        f"{imap_module_directory}/codice/tests/data/"
        f"idle_export_raw.COD_NHK_20230822_122700.csv"
    )
    validation_data = pd.read_csv(validation_file, index_col="SHCOARSE")
    # Remove the timestamp column and data
    if "timestamp" in validation_data.columns:
        validation_data.drop(columns=["timestamp"], inplace=True, errors="ignore")

    # Return the data
    return validation_data


def eu_csv_data():
    """Read EU data from a CSV file and filter the DataFrame to keep only columns
    that match the mnemonics_to_process"""

    # TODO: Pull mnemonics from Telemetry packet "AnalogConversions" instead of hard
    #  coding them here
    mnemonics_to_process = [
        "SPIN_BIN_PERIOD",
        "SPIN_PERIOD",
        "SPIN_PERIOD_TIMER",
        "OPTICS_HV_DAC_ESA_A",
        "OPTICS_HV_DAC_ESA_B",
        "OPTICS_HV_DAC_IONBULK",
        "SENSOR_HV_DAC_SSDO",
        "SENSOR_HV_DAC_SSDB",
        "SENSOR_HV_DAC_APDB",
        "SENSOR_HV_DAC_APDB2",
        "SENSOR_HV_DAC_START_MCP",
        "SENSOR_HV_DAC_STOP_MCP",
        "SENSOR_HV_DAC_STOP_OPTICS_GRID",
        "SBULK_VMON",
        "SSDO_VMON",
        "SSDB_VMON",
        "APDB1_VMON",
        "APDB2_VMON",
        "IOBULK_VMON",
        "ESAA_HI_VMON",
        "SPARE_62",
        "STRMCP_VMON",
        "STPMCP_VMON",
        "STPOG_VMON",
        "APDB1_IMON",
        "ESAB_HI_VMON",
        "SPARE_68",
        "APDB2_IMON",
        "SSDB_IMON",
        "STPMCP_IMON",
        "IOBULK_IMON",
        "STRMCP_IMON",
        "MDM25P_14_T",
        "MDM25P_15_T",
        "MDM25P_16_T",
        "MDM51P_27_T",
        "IO_HVPS_T",
        "LVPS_12V_T",
        "LVPS_5V_T",
        "LVPS_3P3V_T",
        "LVPS_3P3V",
        "LVPS_5V",
        "LVPS_N5V",
        "LVPS_12V",
        "LVPS_N12V",
        "LVPS_3P3V_I",
        "LVPS_5V_I",
        "LVPS_N5V_I",
        "LVPS_12V_I",
        "LVPS_N12V_I",
        "CDH_1P5V",
        "CDH_1P8V",
        "CDH_3P3V",
        "CDH_12V",
        "CDH_N12V",
        "CDH_5V",
        "CDH_5V_ADC",
        "CDH_PROCESSOR_T",
        "CDH_1P8V_LDO_T",
        "CDH_1P5V_LDO_T",
        "CDH_SDRAM_T",
        "SNSR_HVPS_T",
    ]

    filename_eu = "idle_export_eu.COD_NHK_20230822_122700.csv"
    eu_csv = pd.read_csv(filename_eu)

    # Filter the DataFrame to keep only columns that match the mnemonics_to process
    filtered_eu_data = eu_csv[mnemonics_to_process]

    return filtered_eu_data


def validate_unit_conversion(validation_data, filtered_analog):
    """Convert validation data to engineering units using the coefficients from the
    AnalogConversions packet and compare it with filtered_eu_data.

    Parameters
    ----------
    validation_data : pandas DataFrame
        The validation data with raw values
    filtered_analog : pandas DataFrame
        The DataFrame containing coefficients for unit conversion

    Returns
    -------
    engineering_data : pandas DataFrame
        The DataFrame with engineering unit values
    """
    engineering_data = pd.DataFrame(
        np.nan, index=validation_data.index, columns=validation_data.columns
    )

    for mnemonic in validation_data.columns:
        matching_row = filtered_analog[filtered_analog["mnemonic"] == mnemonic]

        if not matching_row.empty:
            coefficients = matching_row[
                ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]
            ].values[0]

            raw_data = validation_data[mnemonic]
            engineering_unit_data = (
                coefficients[0]
                + coefficients[1] * raw_data
                + coefficients[2] * raw_data**2
                + coefficients[3] * raw_data**3
                + coefficients[4] * raw_data**4
                + coefficients[5] * raw_data**5
                + coefficients[6] * raw_data**6
                + coefficients[7] * raw_data**7
            )

            engineering_data[mnemonic] = engineering_unit_data

    return engineering_data


def test_housekeeping_data(decom_test_data, validation_data):
    """Compare the decommutated housekeeping data to the validation data.

    Parameters
    ----------
    decom_test_data : List[Packet]
        The decommuted housekeeping packet data
    validation_data : pandas DataFrame
        The validation data to compare against
    """

    # Take the first decom_packet
    first_data = decom_test_data[0]

    # Get the corresponding row in validation_data based on the "SHCOARSE" value
    validation_row = validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # Compare raw values of housekeeping data
    for key, value in first_data.data.items():
        if key == "SHCOARSE":
            # Compare SHCOARSE value
            assert value.raw_value == validation_row.name
            continue
        # Compare raw values of other housekeeping data
        assert value.raw_value == validation_row[key]


def test_total_packets_in_data_file(decom_test_data):
    """Test if total packets in data file is correct"""
    total_packets = 99
    assert len(decom_test_data) == total_packets


def test_ways_to_get_data(decom_test_data):
    """Test if data can be retrieved using different ways"""
    # First way to get data
    data_value_using_key = decom_test_data[0].data

    # Second way to get data
    data_value_using_list = decom_test_data[0][1]
    # Check if data is same
    assert data_value_using_key == data_value_using_list


def test_unit_conversion(validation_data, filtered_analog):
    """Test if unit conversion is correct"""
    # Call the function to perform unit conversion on the validation data
    engineering_data = validate_unit_conversion(validation_data, filtered_analog)

    # Assert that both DataFrames have the same shape
    assert engineering_data.shape == validation_data.shape

    # Define a tolerance for floating-point precision
    tolerance = 1e-6  # Adjust this tolerance as needed

    # Loop through each column and each row to compare the values
    for column in validation_data.columns:
        for index, row in validation_data.iterrows():
            # Compare each value in the column
            assert abs(engineering_data.at[index, column] - row[column]) < tolerance
