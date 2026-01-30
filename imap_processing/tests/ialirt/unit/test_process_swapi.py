from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swapi import (
    FILLVAL_FLOAT32,
    Consts,
    count_rate,
    geometric_mean,
    optimize_pseudo_parameters,
    process_swapi_ialirt,
)
from imap_processing.swapi.swapi_utils import read_swapi_lut_table
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
def esa_unit_conversion_table() -> pd.DataFrame:
    """
    Read the ESA unit conversion table.

    Returns
    -------
    esa_unit_conversion_table : pandas.DataFrame
        The ESA unit conversion table.
    """
    esa_file_path = (
        imap_module_directory
        / "tests/swapi/lut/imap_swapi_esa-unit-conversion_20250626_v001.csv"
    )
    df = read_swapi_lut_table(esa_file_path)
    return df


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
        f"{imap_module_directory}/tests/ialirt/data/l0/ialirt_test_data.csv"
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
    mock_process_sweep_data,
    xarray_data,
    ialirt_test_data,
    sc_xarray_data,
    esa_unit_conversion_table,
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

    swapi_result = process_swapi_ialirt(xarray_data, esa_unit_conversion_table)

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
    expected_result = 3073.023325893161 * Consts.temporary_density_factor
    assert np.isclose(actual_result, expected_result), (
        f"The actual result of count_rate()"
        f" {actual_result} does not "
        f"match the expected result "
        f"{expected_result}."
    )


def test_optimize_parameters():
    """Test that the optimize_pseudo_parameters() function works correctly."""

    # The following files and values are all validation sets provided by the SWAPI team.
    test_data = {
        "test_set_1": {
            "file_name": "ialirt_test_data_u_sw_550_n_sw_5_T_sw_100000_v2.csv",
            "expected_values": {  # expected output and acceptable tolerance
                "pseudo_speed": (550, 0.01),
                "pseudo_density": (5 / Consts.temporary_density_factor, 0.14),
                "pseudo_temperature": (1e5, 0.25),
            },
        },
        "test_set_2": {
            "file_name": "ialirt_test_data_u_sw_650_n_sw_3.0_T_sw_120000_v2.csv",
            "expected_values": {  # expected output and acceptable tolerance
                "pseudo_speed": (650, 0.01),
                "pseudo_density": (3 / Consts.temporary_density_factor, 0.3),
                "pseudo_temperature": (1.2e5, 0.28),
            },
        },
        "test_set_3": {
            "file_name": "ialirt_test_data_u_sw_400_n_sw_6.0_T_sw_80000_v2.csv",
            "expected_values": {  # expected output and acceptable tolerance
                "pseudo_speed": (400, 0.01),
                "pseudo_density": (6 / Consts.temporary_density_factor, 0.39),
                "pseudo_temperature": (8e4, 0.2),
            },
        },
    }

    calibration_test_file = pd.read_csv(
        f"{imap_module_directory}/tests/ialirt/data/l0/swapi_ialirt_energy_steps.csv"
    )
    energy_passbands = calibration_test_file["Energy"][0:63].to_numpy().astype(float)

    for test_set in test_data:
        energy_data = pd.read_csv(
            f"{imap_module_directory}/tests/ialirt/data/l0/"
            f"{test_data[test_set]['file_name']}",
        )
        count_rates = energy_data["Count Rates [Hz]"].to_numpy()
        count_rates[0] = 0.0
        count_rates_errors = energy_data["Count Rates Error [Hz]"].to_numpy()

        result = optimize_pseudo_parameters(
            count_rates, count_rates_errors, energy_passbands
        )

        result_dict = {
            "pseudo_speed": result[0],
            "pseudo_density": result[1],
            "pseudo_temperature": result[2],
        }

        for param in test_data[test_set]["expected_values"]:
            (
                np.testing.assert_allclose(
                    result_dict[param],
                    test_data[test_set]["expected_values"][param][0],
                    rtol=test_data[test_set]["expected_values"][param][1],
                ),
                f"{param} did not match the expected result within the tolerance.",
            )


def test_optimize_parameters_exception_handling():
    """Test that the optimize_pseudo_parameters() function reports
    speed only when given data that causes curve_fit to fail."""

    expected_speed = 557.279273  # peak passband speed
    file_name = "ialirt_test_data_u_sw_550_n_sw_5_T_sw_100000_v2.csv"

    calibration_test_file = pd.read_csv(
        f"{imap_module_directory}/tests/ialirt/data/l0/swapi_ialirt_energy_steps.csv"
    )
    energy_passbands = calibration_test_file["Energy"][0:63].to_numpy().astype(float)

    energy_data = pd.read_csv(
        f"{imap_module_directory}/tests/ialirt/data/l0/{file_name}"
    )
    count_rates = energy_data["Count Rates [Hz]"].to_numpy()
    count_rates[0] = 0.0
    count_rates = np.tile(count_rates, (2, 1))
    count_rates_errors = energy_data["Count Rates Error [Hz]"].to_numpy()

    """
    code to select the random seed:
    for i in range(100):
    np.random.seed(i)
    result = optimize_pseudo_parameters(count_rates *
    np.abs(np.random.standard_normal(size=count_rates.shape)),
    count_rates_errors, energy_passbands)
    if np.isclose(result['pseudo_speed'][0], expected_speed,
    rtol=1e-6) and np.isnan(result['pseudo_density'][0]):
        print(i)
    """
    np.random.seed(14)
    speed, density, temperature = optimize_pseudo_parameters(
        count_rates * np.abs(np.random.standard_normal(size=count_rates.shape)),
        count_rates_errors,
        energy_passbands,
    )

    np.testing.assert_allclose(speed, expected_speed, rtol=1e-6)
    np.testing.assert_allclose(density, FILLVAL_FLOAT32)
    np.testing.assert_allclose(temperature, FILLVAL_FLOAT32)


def test_optimize_parameters_bad_fit_handling():
    """Test that the optimize_pseudo_parameters() function
    reports speed only when the fit is too poor."""

    file_name = "ialirt_test_data_u_sw_550_n_sw_5_T_sw_100000_v2.csv"

    calibration_test_file = pd.read_csv(
        f"{imap_module_directory}/tests/ialirt/data/l0/swapi_ialirt_energy_steps.csv"
    )
    energy_passbands = calibration_test_file["Energy"][0:63].to_numpy().astype(float)

    energy_data = pd.read_csv(
        f"{imap_module_directory}/tests/ialirt/data/l0/{file_name}"
    )
    count_rates = energy_data["Count Rates [Hz]"].to_numpy()
    count_rates[0] = 0.0
    count_rates_errors = energy_data["Count Rates Error [Hz]"].to_numpy()

    # add high-amplitude randomness to the count rates to make the fit poor
    np.random.seed(0)
    count_rates = count_rates + np.abs(
        np.random.standard_normal(size=count_rates.shape) * count_rates.max()
    )

    speed, density, temperature = optimize_pseudo_parameters(
        count_rates, count_rates_errors, energy_passbands
    )

    expected_speed = (
        np.sqrt(energy_passbands[count_rates.argmax(axis=-1)]) * Consts.speed_coeff
    )

    np.testing.assert_allclose(speed, expected_speed, rtol=1e-6)
    np.testing.assert_allclose(density, FILLVAL_FLOAT32)
    np.testing.assert_allclose(temperature, FILLVAL_FLOAT32)


def test_optimize_parameters_bad_covariance_handling():
    """Test that the optimize_pseudo_parameters() function
    reports speed only when output covariance is nonsensical."""

    file_name = "ialirt_test_data_u_sw_550_n_sw_5_T_sw_100000_v2.csv"

    calibration_test_file = pd.read_csv(
        f"{imap_module_directory}/tests/ialirt/data/l0/swapi_ialirt_energy_steps.csv"
    )
    energy_passbands = calibration_test_file["Energy"][0:63].to_numpy().astype(float)

    energy_data = pd.read_csv(
        f"{imap_module_directory}/tests/ialirt/data/l0/{file_name}"
    )
    count_rates = energy_data["Count Rates [Hz]"].to_numpy()
    count_rates[0] = 0.0
    count_rates_errors = energy_data["Count Rates Error [Hz]"].to_numpy()

    # setting errors to 0 results in infinite covariance
    count_rates_errors *= 0

    speed, density, temperature = optimize_pseudo_parameters(
        count_rates, count_rates_errors, energy_passbands
    )

    expected_speed = (
        np.sqrt(energy_passbands[count_rates.argmax(axis=-1)]) * Consts.speed_coeff
    )

    np.testing.assert_allclose(speed, expected_speed, rtol=1e-6)
    np.testing.assert_allclose(density, FILLVAL_FLOAT32)
    np.testing.assert_allclose(temperature, FILLVAL_FLOAT32)


def test_geometric_mean():
    """Test geometric_mean function."""

    swapi_met_list = [12, 24, 36, 48, 60]

    pseudo_proton_speed_list = [400, 420, 440, 460, 480]
    pseudo_proton_density_list = [5.0, 6.0, 7.0, 8.0, 9.0]
    pseudo_proton_temperature_list = [60000, 62000, 64000, 66000, 68000]

    avg_swapi_met, avg_density, avg_speed, avg_temperature = geometric_mean(
        swapi_met_list,
        pseudo_proton_speed_list,
        pseudo_proton_density_list,
        pseudo_proton_temperature_list,
    )

    expected_density = np.exp(np.mean(np.log(pseudo_proton_density_list)))
    expected_speed = np.exp(np.mean(np.log(pseudo_proton_speed_list)))
    expected_temperature = np.exp(np.mean(np.log(pseudo_proton_temperature_list)))
    expected_met = np.mean(swapi_met_list)

    assert np.isclose(avg_density, expected_density)
    assert np.isclose(avg_speed, expected_speed)
    assert np.isclose(avg_temperature, expected_temperature)
    assert np.isclose(avg_swapi_met, expected_met)


def test_geometric_mean_nan():
    """Test geometric_mean function."""

    swapi_met_list = [12, 24, 36, 48, 60]

    pseudo_proton_speed_list = [400, 420, 440, 460, np.nan]
    pseudo_proton_density_list = [5.0, 6.0, 7.0, 8.0, np.nan]
    pseudo_proton_temperature_list = [60000, 62000, 64000, 66000, np.nan]

    avg_swapi_met, avg_density, avg_speed, avg_temperature = geometric_mean(
        swapi_met_list,
        pseudo_proton_speed_list,
        pseudo_proton_density_list,
        pseudo_proton_temperature_list,
    )

    expected_density = np.exp(np.mean(np.log(pseudo_proton_density_list[0:4])))
    expected_speed = np.exp(np.mean(np.log(pseudo_proton_speed_list[0:4])))
    expected_temperature = np.exp(np.mean(np.log(pseudo_proton_temperature_list[0:4])))
    expected_met = np.mean(swapi_met_list[0:4])

    assert np.isclose(avg_density, expected_density)
    assert np.isclose(avg_speed, expected_speed)
    assert np.isclose(avg_temperature, expected_temperature)
    assert np.isclose(avg_swapi_met, expected_met)


def test_geometric_gaps():
    """Test geometric_mean function."""

    swapi_met_list = [0, 12, 24, 36, 240, 252, 264, 272, 284]

    bool_check = len(swapi_met_list) >= 5 and np.all(
        np.isclose(np.diff(swapi_met_list[-5:]), 12.0, atol=0.05)
    )
    assert not bool_check

    pseudo_proton_speed_list = [400, 420, 440, 460, 480, 500, 520, 540, 560]
    pseudo_proton_density_list = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    pseudo_proton_temperature_list = [
        60000,
        62000,
        64000,
        66000,
        68000,
        70000,
        72000,
        74000,
        76000,
    ]

    avg_swapi_met, avg_density, avg_speed, avg_temperature = geometric_mean(
        swapi_met_list[4::],
        pseudo_proton_speed_list[4::],
        pseudo_proton_density_list[4::],
        pseudo_proton_temperature_list[4::],
    )

    expected_density = np.exp(np.mean(np.log(pseudo_proton_density_list[4::])))
    expected_speed = np.exp(np.mean(np.log(pseudo_proton_speed_list[4::])))
    expected_temperature = np.exp(np.mean(np.log(pseudo_proton_temperature_list[4::])))
    expected_met = np.mean(swapi_met_list[4::])

    assert np.isclose(avg_density, expected_density)
    assert np.isclose(avg_speed, expected_speed)
    assert np.isclose(avg_temperature, expected_temperature)
    assert np.isclose(avg_swapi_met, expected_met)


@pytest.mark.external_test_data
def test_process_spacecraft_packet(
    esa_unit_conversion_table, swapi_postlaunch_sc_packet_path
):
    """Tests spacecraft packet processing."""

    packet_path, xtce_ialirt_path = swapi_postlaunch_sc_packet_path
    xarray_data = tuple(
        packet_file_to_datasets(packet, xtce_ialirt_path, use_derived_value=False)[478]
        for packet in packet_path
    )
    postlaunch_sc_xarray_data = xr.concat(xarray_data, dim="epoch")

    postlaunch_sc_xarray_data["swapi_version"].data = np.full_like(
        postlaunch_sc_xarray_data["swapi_version"].data, 2
    )
    swapi_product = process_swapi_ialirt(
        postlaunch_sc_xarray_data, esa_unit_conversion_table
    )

    assert len(swapi_product) == 0
