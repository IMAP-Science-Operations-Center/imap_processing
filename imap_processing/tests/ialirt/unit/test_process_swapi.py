from unittest import mock

import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swapi import (
    count_rate,
    optimize_pseudo_parameters,
    process_swapi_ialirt,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_swapi_path():
    """Returns the xtce auxiliary directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_swapi.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the binary packet path."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "BinLog CCSDS_FRAG_TLM_20240826_152323Z_IALIRT_data_for_SDC.bin"
    )


@pytest.fixture(scope="session")
def swapi_test_data():
    """Returns the l0 validation dataframe."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "eu_SWP_IAL_20240826_152033.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture
def xarray_data(binary_packet_path, xtce_swapi_path):
    """Create SWAPI xarray dataset for testing."""

    xarray_data = packet_file_to_datasets(
        binary_packet_path, xtce_swapi_path, use_derived_value=True
    )[1187]
    return xarray_data


@pytest.fixture
def sc_xarray_data(sc_packet_path):
    """Extract spacecraft packet for testing."""

    packet_path, xtce_ialirt_path = sc_packet_path
    sc_xarray_data = packet_file_to_datasets(
        packet_path, xtce_ialirt_path, use_derived_value=False
    )[478]
    return sc_xarray_data


@pytest.fixture
def ialirt_test_data():
    """Extract test data for unit tests below."""

    energy_data = pd.read_csv(
        f"{imap_module_directory}/tests/ialirt/test_data/ialirt_test_data.csv"
    )
    count_rates = energy_data["Count Rates [Hz]"].to_numpy()
    count_rates = np.tile(count_rates, (2, 1))
    count_rates_errors = energy_data["Count Rates Error [Hz]"].to_numpy()
    count_rates_errors = np.tile(count_rates_errors, (2, 1))

    return [count_rates, count_rates_errors]


def test_decom_packets(xarray_data, swapi_test_data):
    """Check that all instrument parameters are accounted for after decom."""

    # TODO: confirm w/ SWAPI team validity_enum flag can be
    #  consistent with other instruments.
    fields_to_test = {
        "swapi_flag": "I_ALIRT_STATUS",
        "swapi_reserved": "INST_RES_ST",
        "swapi_seq_number": "SEQ_NUMBER",
        "swapi_version": "SWEEP_TABLE",
        "swapi_coin_cnt0": "COIN_CNT0",
        "swapi_coin_cnt1": "COIN_CNT1",
        "swapi_coin_cnt2": "COIN_CNT2",
        "swapi_coin_cnt3": "COIN_CNT3",
        "swapi_coin_cnt4": "COIN_CNT4",
        "swapi_coin_cnt5": "COIN_CNT5",
        "swapi_spare": "SPARE",
        "swapi_shcoarse": "SHCOARSE",
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


@pytest.mark.external_test_data
@mock.patch("imap_processing.ialirt.l0.process_swapi.process_sweep_data")
def test_process_swapi_ialirt(
    mock_process_sweep_data, xarray_data, ialirt_test_data, sc_xarray_data
):
    """Test that the process_swapi_ialirt() function returns expected keys."""

    mock_process_sweep_data.return_value = ialirt_test_data[0]

    # Adding necessary time variables from spacecraft packet
    xarray_data = xarray_data.assign(sc_sclk_sec=sc_xarray_data["sc_sclk_sec"])
    xarray_data["sc_sclk_sec"].data = sc_xarray_data["sc_sclk_sec"][
        0 : xarray_data["swapi_flag"].shape[0]
    ].data
    xarray_data = xarray_data.assign(sc_sclk_sub_sec=sc_xarray_data["sc_sclk_sub_sec"])
    xarray_data["sc_sclk_sub_sec"].data = sc_xarray_data["sc_sclk_sub_sec"][
        0 : xarray_data["swapi_flag"].shape[0]
    ].data

    swapi_result = process_swapi_ialirt(xarray_data)

    key_names = [
        "apid",
        "met",
        "met_in_utc",
        "ttj2000ns",
        "swapi_pseudo_proton_density",
        "swapi_pseudo_proton_speed",
        "swapi_pseudo_proton_temperature",
    ]

    for key in key_names:
        assert swapi_result[0][key] is not None, (
            f"The expected attribute {key} was not filled in the result dict."
        )


def test_count_rate():
    """Use random realistic values to test for expected output of count_rate()."""

    actual_result = count_rate(1370, *[550, 5.27, 1e5])
    expected_result = 621.0028766348703
    assert actual_result == expected_result, (
        f"The actual result of count_rate()"
        f" {actual_result} does not "
        f"match the expected result "
        f"{expected_result}."
    )


@pytest.mark.skip(reason="Differences between scipy versions.")
def test_optimize_parameters(xarray_data, ialirt_test_data):
    """Test that the optimize_pseudo_parameters() function works correctly."""

    result = optimize_pseudo_parameters(*ialirt_test_data)

    # Test output corresponding to this exact set of test inputs.
    expected_speed = [550.2067500045512, 550.2067500045512]
    expected_density = [15.964441588773008, 15.964441588773008]
    expected_temperature = [101695.2160638631, 101695.2160638631]

    assert np.allclose(result["pseudo_speed"], expected_speed, rtol=0.01), (
        "Pseudo speed did not match the expected result."
    )
    assert np.allclose(result["pseudo_density"], expected_density, rtol=0.01), (
        "Pseudo density did not match the expected result."
    )
    assert np.allclose(result["pseudo_temperature"], expected_temperature, rtol=0.01), (
        "Pseudo temperature did not match the expected result."
    )


@pytest.mark.external_test_data
def test_process_spacecraft_packet(sc_xarray_data):
    """Tests spacecraft packet processing."""

    # Case 1: Not fixing the sequence number attribute, which is all zeros.
    swapi_product = process_swapi_ialirt(sc_xarray_data)
    assert swapi_product == []

    # Case 2: Overwriting swapi_seq_number to be an acceptable array of numbers.
    # Calculate how many times to tile the sequence to reach length of sc packet
    target_length = sc_xarray_data["swapi_seq_number"].shape[0]
    base_sequence = np.arange(12)
    repeat_times = (target_length // len(base_sequence)) + 1  # Over-repeat

    # Tile the sequence and truncate to target_length
    extended_data = np.tile(base_sequence, repeat_times)[:target_length]
    sc_xarray_data["swapi_seq_number"].data = extended_data

    swapi_product1 = process_swapi_ialirt(sc_xarray_data)
    key_names = [
        "apid",
        "met",
        "met_in_utc",
        "ttj2000ns",
        "swapi_pseudo_proton_density",
        "swapi_pseudo_proton_speed",
        "swapi_pseudo_proton_temperature",
    ]

    for key in key_names:
        assert swapi_product1[0][key] is not None, (
            f"The expected attribute {key} was not filled in the result dict."
        )
