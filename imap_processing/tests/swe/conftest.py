import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.utils.swe_utils import SWEAPID
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    packet_file = (
        imap_module_directory / "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    )
    xtce_document = (
        imap_module_directory / "swe/packet_definitions/swe_packet_definition.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file, xtce_document, use_derived_value=False
    )
    return datasets_by_apid[SWEAPID.SWE_SCIENCE]


@pytest.fixture(scope="session")
def decom_test_data_derived():
    """Read test data from file"""
    packet_file = (
        imap_module_directory / "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    )
    xtce_document = (
        imap_module_directory / "swe/packet_definitions/swe_packet_definition.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file, xtce_document, use_derived_value=True
    )
    return datasets_by_apid[SWEAPID.SWE_SCIENCE]


@pytest.fixture(scope="session")
def l1a_validation_df():
    """Read validation data from file"""
    l1_val_path = imap_module_directory / "tests/swe/l1_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L1A.dat"

    # Define column names for validation data
    column_names = [
        "shcoarse",
        "raw_cnt_cem_1",
        "raw_cnt_cem_2",
        "raw_cnt_cem_3",
        "raw_cnt_cem_4",
        "raw_cnt_cem_5",
        "raw_cnt_cem_6",
        "raw_cnt_cem_7",
        "decom_cnt_cem_1",
        "decom_cnt_cem_2",
        "decom_cnt_cem_3",
        "decom_cnt_cem_4",
        "decom_cnt_cem_5",
        "decom_cnt_cem_6",
        "decom_cnt_cem_7",
    ]

    # Read the data, specifying na_values and delimiter
    df = pd.read_csv(
        l1_val_path / filename,
        skiprows=10,  # Skip the first 10 rows of comments
        sep=r"\s*,\s*",  # Regex to handle spaces and commas as delimiters
        names=column_names,
        na_values=["", " "],  # Treat empty strings or spaces as NaN
        engine="python",
    )

    # Fill NaNs with the previous value
    df["shcoarse"] = df["shcoarse"].ffill()
    return df


@pytest.fixture(scope="session")
def l1b_validation_df():
    """Read validation data from file"""
    l1_val_path = imap_module_directory / "tests/swe/l1_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L1B_v3.dat"

    # Define column names for validation data
    column_names = [
        "shcoarse",
        "cem_1",
        "cem_2",
        "cem_3",
        "cem_4",
        "cem_5",
        "cem_6",
        "cem_7",
    ]

    # Read the data, specifying na_values and delimiter
    df = pd.read_csv(
        l1_val_path / filename,
        skiprows=12,  # Skip the first 10 rows of comments
        sep=r"\s*,\s*",  # Regex to handle spaces and commas as delimiters
        names=column_names,
        na_values=["", " "],  # Treat empty strings or spaces as NaN
        engine="python",
    )

    # Fill NaNs with the previous value
    df["shcoarse"] = df["shcoarse"].ffill()
    return df
