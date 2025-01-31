import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_swapi_path():
    """Returns the xtce auxiliary directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_swapi.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the xtce directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "BinLog CCSDS_FRAG_TLM_20240826_152323Z_IALIRT_data_for_SDC.bin"
    )


@pytest.fixture(scope="session")
def swapi_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "eu_SWP_IAL_20240826_152033.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture()
def xarray_data(binary_packet_path, xtce_swapi_path):
    """Create xarray data"""
    apid = 1187

    xarray_data = packet_file_to_datasets(
        binary_packet_path, xtce_swapi_path, use_derived_value=True
    )[apid]
    return xarray_data


def test_decom_packets(xarray_data, swapi_test_data):
    """This function checks that all instrument parameters are accounted for."""

    # TODO: confirm w/ SWAPI team validity_enum flag can be
    #  consistent with other instruments.
    fields_to_test = {
        "swapi_flag": "I_ALIRT_STATUS",
        "swapi_reserved": "INST_RES_ST",
        "swapi_seq": "SEQ_NUMBER",
        "swapi_version": "SWEEP_TABLE",
        "swapi_coin_1": "COIN_CNT0",
        "swapi_coin_2": "COIN_CNT1",
        "swapi_coin_3": "COIN_CNT2",
        "swapi_coin_4": "COIN_CNT3",
        "swapi_coin_5": "COIN_CNT4",
        "swapi_coin_6": "COIN_CNT5",
        "swapi_spare": "SPARE",
    }
    _, index, test_index = np.intersect1d(
        xarray_data["swapi_acq"], swapi_test_data["ACQ_TIME"], return_indices=True
    )

    for xarray_field, test_field in fields_to_test.items():
        actual_values = xarray_data[xarray_field].values[index]
        expected_values = swapi_test_data[test_field].values[test_index]

        # Assert that all values match
        assert np.all(actual_values == expected_values), (
            f"Mismatch found in {xarray_field}: "
            f"actual {actual_values}, expected {expected_values}"
        )
