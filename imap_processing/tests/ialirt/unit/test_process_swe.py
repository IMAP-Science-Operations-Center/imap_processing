import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_swe_path():
    """Returns the xtce auxiliary directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_swe.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the xtce directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "20240827095047_SWE_IALIRT_packet.bin"
    )


@pytest.fixture(scope="session")
def swe_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "idle_export_eu.SWE_IALIRT_20240827_093852.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture()
def xarray_data(binary_packet_path, xtce_swe_path):
    """Create xarray data"""
    apid = 1360

    xarray_data = packet_file_to_datasets(
        binary_packet_path, xtce_swe_path, use_derived_value=True
    )[apid]
    return xarray_data


def test_decom_packets(xarray_data, swe_test_data):
    """This function checks that all instrument parameters are accounted for."""

    fields_to_test = {
        "swe_shcoarse": "SHCOARSE",
        "swe_acq_sec": "ACQUISITION_TIME",
        "swe_acq_sub": "ACQUISITION_TIME_SUBSECOND",
        "swe_nom_flag": "INSTRUMENT_NOMINAL_FLAG",
        "swe_ops_flag": "SCIENCE_OPERATION_FLAG",
        "swe_seq": "SEQUENCE_NUMBER",
        "swe_cem1_e1": "ELEC_COUNTS_SPIN_I_POL_0_E_0J",
        "swe_cem1_e2": "ELEC_COUNTS_SPIN_I_POL_0_E_1J",
        "swe_cem1_e3": "ELEC_COUNTS_SPIN_I_POL_0_E_2J",
        "swe_cem1_e4": "ELEC_COUNTS_SPIN_I_POL_0_E_3J",
        "swe_cem2_e1": "ELEC_COUNTS_SPIN_I_POL_1_E_0J",
        "swe_cem2_e2": "ELEC_COUNTS_SPIN_I_POL_1_E_1J",
        "swe_cem2_e3": "ELEC_COUNTS_SPIN_I_POL_1_E_2J",
        "swe_cem2_e4": "ELEC_COUNTS_SPIN_I_POL_1_E_3J",
        "swe_cem3_e1": "ELEC_COUNTS_SPIN_I_POL_2_E_0J",
        "swe_cem3_e2": "ELEC_COUNTS_SPIN_I_POL_2_E_1J",
        "swe_cem3_e3": "ELEC_COUNTS_SPIN_I_POL_2_E_2J",
        "swe_cem3_e4": "ELEC_COUNTS_SPIN_I_POL_2_E_3J",
        "swe_cem4_e1": "ELEC_COUNTS_SPIN_I_POL_3_E_0J",
        "swe_cem4_e2": "ELEC_COUNTS_SPIN_I_POL_3_E_1J",
        "swe_cem4_e3": "ELEC_COUNTS_SPIN_I_POL_3_E_2J",
        "swe_cem4_e4": "ELEC_COUNTS_SPIN_I_POL_3_E_3J",
        "swe_cem5_e1": "ELEC_COUNTS_SPIN_I_POL_4_E_0J",
        "swe_cem5_e2": "ELEC_COUNTS_SPIN_I_POL_4_E_1J",
        "swe_cem5_e3": "ELEC_COUNTS_SPIN_I_POL_4_E_2J",
        "swe_cem5_e4": "ELEC_COUNTS_SPIN_I_POL_4_E_3J",
        "swe_cem6_e1": "ELEC_COUNTS_SPIN_I_POL_5_E_0J",
        "swe_cem6_e2": "ELEC_COUNTS_SPIN_I_POL_5_E_1J",
        "swe_cem6_e3": "ELEC_COUNTS_SPIN_I_POL_5_E_2J",
        "swe_cem6_e4": "ELEC_COUNTS_SPIN_I_POL_5_E_3J",
        "swe_cem7_e1": "ELEC_COUNTS_SPIN_I_POL_6_E_0J",
        "swe_cem7_e2": "ELEC_COUNTS_SPIN_I_POL_6_E_1J",
        "swe_cem7_e3": "ELEC_COUNTS_SPIN_I_POL_6_E_2J",
        "swe_cem7_e4": "ELEC_COUNTS_SPIN_I_POL_6_E_3J",
    }
    _, index, test_index = np.intersect1d(
        xarray_data["swe_shcoarse"], swe_test_data["SHCOARSE"], return_indices=True
    )

    for xarray_field, test_field in fields_to_test.items():
        actual_values = xarray_data[xarray_field].values[index]
        expected_values = swe_test_data[test_field].values[test_index]

        # Assert that all values match
        assert np.all(actual_values == expected_values), (
            f"Mismatch found in {xarray_field}: "
            f"actual {actual_values}, expected {expected_values}"
        )
