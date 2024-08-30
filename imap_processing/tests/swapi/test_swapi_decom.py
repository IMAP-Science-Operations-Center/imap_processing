import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.swapi.l1.swapi_l1 import (
    SWAPIAPID,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def decom_test_science_data(swapi_l0_test_data_path):
    """Read test data from file with derived values"""
    test_file = "imap_swapi_l0_raw_20231012_v001.pkts"
    packet_file = imap_module_directory / swapi_l0_test_data_path / test_file
    packet_definition = (
        f"{imap_module_directory}/swapi/packet_definitions/swapi_packet_definition.xml"
    )
    return packet_file_to_datasets(
        packet_file, packet_definition, use_derived_value=True
    )


@pytest.fixture(scope="session")
def decom_test_data(swapi_l0_test_data_path):
    """Read test data from file as raw values"""
    test_file = "imap_swapi_l0_raw_20231012_v001.pkts"
    packet_file = imap_module_directory / swapi_l0_test_data_path / test_file
    packet_definition = (
        f"{imap_module_directory}/swapi/packet_definitions/swapi_packet_definition.xml"
    )
    return packet_file_to_datasets(
        packet_file, packet_definition, use_derived_value=False
    )


def test_number_of_packets(decom_test_data):
    """This test and validate number of packets."""
    sci_packets = decom_test_data[SWAPIAPID.SWP_SCI]
    expected_sci_packets = 54
    assert len(sci_packets["epoch"]) == expected_sci_packets

    hk_packets = decom_test_data[SWAPIAPID.SWP_HK]
    expected_hk_packets = 54
    assert len(hk_packets["epoch"]) == expected_hk_packets


def test_swapi_sci_data(decom_test_science_data, swapi_l0_validation_data_path):
    """This test and validate raw data of SWAPI raw science data."""
    # read validation data
    raw_validation_data = pd.read_csv(
        swapi_l0_validation_data_path / "idle_export_eu.SWP_SCI_20231012_125245.csv",
        index_col="SHCOARSE",
    )

    sci_packets = decom_test_science_data[SWAPIAPID.SWP_SCI]
    first_data = sci_packets.isel(epoch=0)
    validation_data = raw_validation_data.loc[first_data["shcoarse"].values]

    # compare raw values of validation data
    for key in raw_validation_data.columns:
        if key in [
            "PHAPID",
            "timestamp",
            "PHGROUPF",
            "PHSHF",
            "PHVERNO",
            "PHSEQCNT",
            "PHDLEN",
            "PHTYPE",
        ]:
            continue

        # for SHCOARSE we need the name of the column.
        # This is done because pandas removed it from the
        # main columns to make it the index.
        assert first_data[key.lower()].values == (
            validation_data[key] if key != "SHCOARSE" else validation_data.name
        )


def test_swapi_hk_data(decom_test_data, swapi_l0_validation_data_path):
    """This test and validate raw data of SWAPI raw housekeeping data."""
    # read validation data
    raw_validation_data = pd.read_csv(
        swapi_l0_validation_data_path / "idle_export_raw.SWP_HK_20231012_125245.csv",
        index_col="SHCOARSE",
    )

    hk_packets = decom_test_data[SWAPIAPID.SWP_HK]
    first_data = hk_packets.isel(epoch=0)
    validation_data = raw_validation_data.loc[first_data["shcoarse"].values]

    # compare raw values of validation data
    for key in raw_validation_data.columns:
        if key in [
            "PHAPID",
            "timestamp",
            "PHGROUPF",
            "PHSHF",
            "PHVERNO",
            "PHSEQCNT",
            "PHDLEN",
            "PHTYPE",
        ]:
            continue
        # for SHCOARSE we need the name of the column.
        # This is done because pandas removed it from the
        # main columns to make it the index.
        assert first_data[key.lower()].values == (
            validation_data[key] if key != "SHCOARSE" else validation_data.name
        )
