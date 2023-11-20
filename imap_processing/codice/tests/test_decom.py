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


@pytest.fixture(scope="session")
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

    filename_eu = Path(
        f"{imap_module_directory}/codice/tests/data/idle_export_eu"
        f".COD_NHK_20230822_122700 2.csv"
    )
    eu_csv = pd.read_csv(filename_eu)

    # Filter the DataFrame to keep only columns that match the mnemonics_to process
    filtered_eu_data = eu_csv[mnemonics_to_process]

    return filtered_eu_data


@pytest.fixture(scope="session")
def read_and_filter_analog_data():
    """Read and filter analog data from an Excel file.

    Returns
    -------
    filtered_analog_data : pandas DataFrame
        The filtered analog data
    """

    # TODO: Change the path to the Excel file once the new TEL document is updated
    excel_file_path = Path(
        "/Users/gamo6782/Desktop/repos/imap_processing/tools"
        "/xtce_generation/TLM_COD.xlsx"
    )

    # Read analog data from an Excel file into a pandas DataFrame
    analog = pd.read_excel(excel_file_path, sheet_name="AnalogConversions")

    # Filterd analog data to keep only rows with 'packetName' as 'COD_NHK'
    filtered_analog = analog[analog["packetName"] == "COD_NHK"].copy()

    # List of columns to remove
    columns_to_remove = [
        "packetName",
        "convertAs",
        "segNumber",
        "lowValue",
        "highValue",
    ]

    # Drop the specified columns
    filtered_analog.drop(columns=columns_to_remove, inplace=True)

    # Convert all string columns to uppercase
    # TODO: This will be replaced once the new TEL document is updated
    filtered_analog = filtered_analog.applymap(
        lambda x: x.upper() if isinstance(x, str) else x
    )

    return filtered_analog


@pytest.fixture(scope="session")
def validate_unit_conversion(validation_data, filtered_analog_data):
    """Convert validation data to engineering units using the coefficients from the
    AnalogConversions packet and compare it with filtered_analog_data.

    Parameters
    ----------
    validation_data : pandas DataFrame
        The validation data with raw values
    filtered_analog_data : pandas DataFrame
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
        matching_row = filtered_analog_data[filtered_analog_data.index == mnemonic]

        if not matching_row.empty:
            coefficients = matching_row[
                ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]
            ].values[0]

            raw_data = validation_data[mnemonic]

            # Handle NaN values by excluding them from the calculations
            mask = ~np.isnan(raw_data)

            # Create a polynomial using the coefficients
            poly = np.polynomial.Polynomial(coefficients)

            # Evaluate the polynomial on the raw data
            engineering_unit_data = poly(raw_data[mask])

            # Assign the result to the engineering_data DataFrame
            engineering_data.loc[mask, mnemonic] = engineering_unit_data

            print(f"Converted {mnemonic} to engineering units")

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


# TODO: Add test for unit conversion of engineering data and eu data
def test_unit_conversion_coefficients(filtered_analog_data, validation_data):
    """Test if the coefficients in engineering data match the units in EU data
    by mnemonics.

    Parameters
    ----------
    filtered_analog_data : pandas DataFrame
        The DataFrame containing coefficients for unit conversion
    validation_data : pandas DataFrame
        The validation data with raw values
    """
    # TODO: Call the function to perform unit conversion
    engineering_data = validate_unit_conversion(validation_data, filtered_analog_data)

    # Ensure that the engineering data has the same mnemonics as filtered_eu_data
    assert set(engineering_data.columns) == set(filtered_analog_data.index)

    # Check if coefficients in engineering data match the units in EU data by mnemonics
    for mnemonic in engineering_data.columns:
        # Get the corresponding row in filtered_analog_data based on the mnemonic
        matching_row = filtered_analog_data[filtered_analog_data.index == mnemonic]

        # Check if the row is not empty
        assert not matching_row.empty

        # Extract coefficients from filtered_analog_data
        coefficients_from_filtered_analog = matching_row[
            ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]
        ].values[0]

        # Extract coefficients from engineering_data
        coefficients_from_engineering_data = engineering_data.loc[:, mnemonic].values

        # Check if the coefficients match
        np.testing.assert_array_almost_equal(
            coefficients_from_engineering_data, coefficients_from_filtered_analog
        )
