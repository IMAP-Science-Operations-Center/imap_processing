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
        binary_packet_path, xtce_swapi_path, use_derived_value=False
    )[apid]
    return xarray_data


def test_decom_packets(xarray_data, swapi_test_data):
    """This function checks that all instrument parameters are accounted for."""

    swapi_acq = xarray_data["swapi_acq"]
    swapi_flag = xarray_data["swapi_flag"]
    swapi_reserved = xarray_data["swapi_reserved"]
    swapi_seq = xarray_data["swapi_seq"]
    swapi_version = xarray_data["swapi_version"]
    swapi_coin_1 = xarray_data["swapi_coin_1"]
    swapi_coin_2 = xarray_data["swapi_coin_2"]
    swapi_coin_3 = xarray_data["swapi_coin_3"]
    swapi_coin_4 = xarray_data["swapi_coin_4"]
    swapi_coin_5 = xarray_data["swapi_coin_5"]
    swapi_coin_6 = xarray_data["swapi_coin_6"]
    swapi_spare = xarray_data["swapi_spare"]

    expected_swapi_acq = swapi_test_data["ACQ_TIME"]
    expected_swapi_flag = swapi_test_data["I_ALIRT_STATUS"]
    expected_swapi_reserved = swapi_test_data["INST_RES_ST"]
    expected_swapi_seq = swapi_test_data["SEQ_NUMBER"]
    expected_swapi_version = swapi_test_data["SWEEP_TABLE"]
    expected_swapi_coin_1 = swapi_test_data["COIN_CNT0"]
    expected_swapi_coin_2 = swapi_test_data["COIN_CNT1"]
    expected_swapi_coin_3 = swapi_test_data["COIN_CNT2"]
    expected_swapi_coin_4 = swapi_test_data["COIN_CNT3"]
    expected_swapi_coin_5 = swapi_test_data["COIN_CNT4"]
    expected_swapi_coin_6 = swapi_test_data["COIN_CNT5"]
    expected_swapi_spare = swapi_test_data["SPARE"]

    matching_indices = np.nonzero(
        np.isin(xarray_data["swapi_acq"], swapi_test_data["ACQ_TIME"])
    )[0]
    assert np.all(swapi_flag[matching_indices] == expected_swapi_flag)

    print("hi")
