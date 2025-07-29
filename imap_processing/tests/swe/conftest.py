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


def read_validation_df(val_filepath, column_names, skiprows):
    """Read and return validation dataframe from file."""
    df = pd.read_csv(
        val_filepath,
        skiprows=skiprows,
        sep=r"\s*,\s*",
        names=column_names,
        na_values=["", " "],
        engine="python",
    )
    df["shcoarse"] = df["shcoarse"].ffill()
    return df


@pytest.fixture(scope="session")
def l1a_validation_df():
    """Read validation data from file"""
    l1_val_path = imap_module_directory / "tests/swe/l1_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L1A.dat"
    l1a_val_path = l1_val_path / filename
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

    return read_validation_df(l1a_val_path, column_names, skiprows=10)


@pytest.fixture(scope="session")
def l1b_validation_df():
    """Read validation data from file"""
    l1_val_path = imap_module_directory / "tests/swe/l1_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L1B_v3.dat"
    l1b_val_path = l1_val_path / filename
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

    return read_validation_df(l1b_val_path, column_names, skiprows=12)


@pytest.fixture(scope="session")
def l2_sector_validation_df():
    """Validation for phase_space_density_spin_sector variable in L2 data"""
    l2_sector_val = imap_module_directory / "tests/swe/l2_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_v0B_15.dat"
    l2_val_path = l2_sector_val / filename
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

    return read_validation_df(l2_val_path, column_names, skiprows=14)


@pytest.fixture(scope="session")
def l2_binned_flux_validation_df():
    """Validation data for flux variable in L2 data.

    This is for 15 spin period data.
    """
    l2_val_path = imap_module_directory / "tests/swe/l2_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_bins_v1F_15.dat"
    l2_val_path = l2_val_path / filename
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

    return read_validation_df(l2_val_path, column_names, skiprows=13)


@pytest.fixture(scope="session")
def l2_binned_psd_validation_df():
    """Validation for phase_space_density variable in L2 data

    This is for 15 spin period data.
    """
    l2_binned_psd_val_path = imap_module_directory / "tests/swe/l2_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_bins_v0F_15.dat"

    l2_val_path = l2_binned_psd_val_path / filename
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

    return read_validation_df(l2_val_path, column_names, skiprows=13)


@pytest.fixture(scope="session")
def l2_binned_flux_14sec_validation_df():
    """Validation data for flux variable in L2 data

    This is for 14.6 spin period data.
    """
    l2_val_path = imap_module_directory / "tests/swe/l2_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_bins_v1H_14_6.dat"
    l2_val_path = l2_val_path / filename
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

    return read_validation_df(l2_val_path, column_names, skiprows=13)


@pytest.fixture(scope="session")
def l2_binned_psd_14sec_validation_df():
    """Validation data for phase_space_density variable in L2 data.

    This is for 14.6 spin period data.
    """
    l2_val_path = imap_module_directory / "tests/swe/l2_validation"
    filename = "swe_l0_unpacked-data_20240510_v001_VALIDATION_L2_bins_v0H_14_6.dat"
    l2_val_path = l2_val_path / filename
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

    return read_validation_df(l2_val_path, column_names, skiprows=13)
