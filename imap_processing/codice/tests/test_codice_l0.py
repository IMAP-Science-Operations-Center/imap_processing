"""Tests the decommutation process for CoDICE CCSDS Packets. This also tests the
unit conversion process for CoDICE housekeeping data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import space_packet_parser
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.codice import codice_l0
from imap_processing.codice.utils import create_dataset
from imap_processing.utils import convert_raw_to_eu


@pytest.fixture(scope="session")
def decom_test_data() -> list:
    """Read test data from file

    Returns
    -------
    data_packet_list : list[space_packet_parser.parser.Packet]
        The list of decommutated packets
    """

    packet_file = Path(
        f"{imap_module_directory}/codice/tests/data/"
        f"raw_ccsds_20230822_122700Z_idle.bin"
    )
    Path(f"{imap_module_directory}/codice/packet_definitions/P_COD_NHK.xml")
    data_packet_list = codice_l0.decom_packets(packet_file)
    data_packet_list = [
        packet
        for packet in data_packet_list
        if packet.header["PKT_APID"].raw_value == 1136
    ]

    return data_packet_list


@pytest.fixture(scope="session")
def validation_data() -> pd.core.frame.DataFrame:
    """Read in validation data from the CSV file

    Returns
    -------
    validation_data : pandas.core.frame.DataFrame
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

    dataset = xr.Dataset()
    dataset["First_mnemonic"] = xr.DataArray(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    # Return the data
    return validation_data


def test_housekeeping_data(
    decom_test_data: list[space_packet_parser.parser.Packet],
    validation_data: pd.core.frame.DataFrame,
):
    """Compare the decommutated housekeeping data to the validation data.

    Parameters
    ----------
    decom_test_data : list[space_packet_parser.parser.Packet]
        The decommutated housekeeping packet data
    validation_data : pandas.core.frame.DataFrame
        The validation data to compare against
    """

    l1a_hk_ds = create_dataset(decom_test_data)
    eu_hk_data = convert_raw_to_eu(
        l1a_hk_ds,
        imap_module_directory / "codice/tests/data/eu_unit_lookup_table.csv",
        "P_COD_NHK",
    )
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

    # Compare EU values of housekeeping data
    for field in eu_hk_data:
        assert round(eu_hk_data[field].data[1], 5) == round(validation_data[field], 5)


def test_total_packets_in_data_file(
    decom_test_data: list[space_packet_parser.parser.Packet],
):
    """Test if total packets in data file is correct

    Parameters
    ----------
    decom_test_data : list[space_packet_parser.parser.Packet]
        The decommutated housekeeping packet data
    """

    total_packets = 99
    assert len(decom_test_data) == total_packets


def test_ways_to_get_data(decom_test_data: list[space_packet_parser.parser.Packet]):
    """Test if data can be retrieved using different ways

    Parameters
    ----------
    decom_test_data : list[space_packet_parser.parser.Packet]
        The decommutated housekeeping packet data
    """

    # First way to get data
    data_value_using_key = decom_test_data[0].data

    # Second way to get data
    data_value_using_list = decom_test_data[0][1]

    # Check if data is same
    assert data_value_using_key == data_value_using_list
