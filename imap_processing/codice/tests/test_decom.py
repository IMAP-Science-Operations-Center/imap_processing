"""Tests the decommutation process for CoDICE CCSDS Packets. This also tests the
unit conversion process for CoDICE housekeeping data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
def raw_data():
    """Read in validation data from the CSV file

    Returns
    -------
    raw_data : pandas DataFrame
        The raw data read from the CSV, cleaned up and ready to compare
        the decommutated packet with
    """

    # Read in the CSV file (perhaps to a pandas dataframe?)
    validation_file = Path(
        f"{imap_module_directory}/codice/tests/data/"
        f"idle_export_raw.COD_NHK_20230822_122700.csv"
    )
    raw_data = pd.read_csv(validation_file, index_col="SHCOARSE")
    # Remove the timestamp column and data
    if "timestamp" in raw_data.columns:
        raw_data.drop(columns=["timestamp"], inplace=True, errors="ignore")

    dataset = xr.Dataset()
    dataset["First_mnemonic"] = xr.DataArray(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    # Return the data
    return raw_data


def convert_raw_to_eu(dataset: xr.Dataset, packet_name):
    """Convert raw data to engineering unit.

    Parameters
    ----------
    dataset : xr.Dataset
        Raw data.
    packet_name: str
        Packet name

    Returns
    -------
    xr.Dataset
        Raw data converted to engineering unit as needed.
    """
    print(f"Type of 'dataset': {type(dataset)}")  # Debugging line

    conversion_table_path = (
        Path(imap_module_directory) / "codice/tests/data/eu_unit_lookup_table.csv"
    )

    # Make sure there is a column called "index" with unique
    # value such as 0, 1, 2, 3, ...
    eu_conversion_table = pd.read_csv(
        conversion_table_path,
        index_col="index",
    )

    # Look up all metadata fields for the packet name
    metadata_list = eu_conversion_table.loc[
        eu_conversion_table["packetName"] == packet_name
    ]

    # for each metadata field, convert raw value to engineering unit
    for field in metadata_list.index:
        metadata_field = metadata_list.loc[field]["mnemonic"]

        # Debugging line
        print(f"Processing metadata field: {metadata_field}")

        # Check if the metadata_field is present in the dataset
        if metadata_field in dataset:
            # On this line, we are getting the coefficients from the
            # table and then reverse them because np.polyval is
            # expecting coefficients in descending order
            coeff_values = metadata_list.loc[
                metadata_list["mnemonic"] == metadata_field
            ].values[0][6:][::-1]

            # Convert the raw value to engineering unit
            dataset[metadata_field].data = np.polyval(
                coeff_values, dataset[metadata_field].data
            )
        else:
            print(f"Warning: Metadata field {metadata_field} not found in the dataset.")

    return dataset


def test_housekeeping_data(decom_test_data, raw_data):
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
    validation_row = raw_data.loc[first_data.data["SHCOARSE"].raw_value]

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


def test_derived_eu_data(decom_test_data, raw_data):
    """Test the derived engineering unit data.

    Parameters
    ----------
    decom_test_data : List[Packet]
        The decommuted housekeeping packet data
    raw_data : pandas DataFrame
        The raw validation data
    """

    # Assuming 'First_mnemonic' is the first derived EU mnemonic, modify accordingly
    derived_eu_data_key = "First_mnemonic"

    # Take the first decom_packet
    first_data = decom_test_data[0]

    # Check if the derived EU data is present in the decommuted data
    assert derived_eu_data_key in first_data.data

    # Get the corresponding value from the raw validation data
    validation_row = raw_data.loc[first_data.data["SHCOARSE"].raw_value]
    validation_value = validation_row[derived_eu_data_key]

    # Compare the derived EU data values
    assert first_data.data[derived_eu_data_key].raw_value == validation_value
