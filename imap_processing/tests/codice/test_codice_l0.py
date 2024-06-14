"""Tests the decommutation process for CoDICE CCSDS Packets. This also tests the
unit conversion process for CoDICE housekeeping data."""

from pathlib import Path

import pandas as pd
import pytest
import space_packet_parser

from imap_processing import imap_module_directory
from imap_processing.codice import codice_l0
from imap_processing.codice.utils import create_hskp_dataset
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
        f"{imap_module_directory}/tests/codice/data/"
        f"imap_codice_l0_hskp_20100101_v001.pkts"
    )

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

    # Read in the CSV file
    validation_file = Path(
        f"{imap_module_directory}/tests/codice/data/"
        f"idle_export_raw.COD_NHK_20230822_122700.csv"
    )
    validation_data = pd.read_csv(validation_file, index_col="SHCOARSE")

    if "timestamp" in validation_data.columns:
        validation_data.drop(columns=["timestamp"], inplace=True, errors="ignore")

    return validation_data


def test_eu_hk_data(
    decom_test_data: list[space_packet_parser.parser.Packet],
    validation_data: pd.core.frame.DataFrame,
):
    """Compare the engineering unit (EU) housekeeping data to the validation data.

    Parameters
    ----------
    decom_test_data : list[space_packet_parser.parser.Packet]
        The decommutated housekeeping packet data
    validation_data : pandas.core.frame.DataFrame
        The validation data to compare against
    """

    l1a_hk_ds = create_hskp_dataset(decom_test_data, "001")
    eu_hk_data = convert_raw_to_eu(
        l1a_hk_ds,
        imap_module_directory / "tests/codice/data/eu_unit_lookup_table.csv",
        "P_COD_NHK",
    )
    first_data = decom_test_data[0]
    validation_row = validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # Determine the number of CCSDS header fields (7 is standard)
    num_ccsds_header_fields = 7

    # Compare EU values of housekeeping data, skipping CCSDS header fields
    for idx, field in enumerate(eu_hk_data):
        # Skip the first num_ccsds_header_fields fields
        if idx < num_ccsds_header_fields:
            continue
        # Skip SHCOARSE
        if field == "SHCOARSE":
            continue

        eu_values = eu_hk_data[field].data
        validation_values = validation_row[field]

        # Compare each individual element
        for eu_val, validation_val in zip(eu_values, [validation_values]):
            assert round(eu_val, 5) == round(validation_val, 5)


def test_raw_hk_data(
    decom_test_data: list[space_packet_parser.parser.Packet],
    validation_data: pd.core.frame.DataFrame,
):
    """Compare the raw housekeeping data to the validation data.

    Parameters
    ----------
    decom_test_data : list[space_packet_parser.parser.Packet]
        The decommutated housekeeping packet data
    validation_data : pandas.core.frame.DataFrame
        The validation data to compare against
    """

    first_data = decom_test_data[0]
    validation_row = validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # Compare raw values of housekeeping data
    for key, value in first_data.data.items():
        if key == "SHCOARSE":
            assert value.raw_value == validation_row.name
            continue
        assert value.raw_value == validation_row[key]


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

    data_value_using_key = decom_test_data[0].data
    data_value_using_list = decom_test_data[0][1]

    # Check if data is same
    assert data_value_using_key == data_value_using_list
