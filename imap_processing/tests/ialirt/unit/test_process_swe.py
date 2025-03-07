from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swe import process_swe, decompress_counts, prepare_raw_counts, phi_to_bin
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


@pytest.fixture()
def fields_to_test():
    """Create a dictionary to convert names"""
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
    return fields_to_test

# TODO: double check this test
@pytest.fixture()
def grouped_data():
    """Creates grouped data for prepare_raw_counts test."""
    epoch = np.arange(60)

    group = np.zeros(60, dtype=np.int32)
    data_vars = {"group": ("epoch", group)}

    for cem in range(1, 8):
        for e in range(1, 5):
            key = f"swe_cem{cem}_e{e}"
            data_vars[key] = ("epoch", np.full(60, cem * 10 + e, dtype=np.uint8))

    grouped_data = xr.Dataset(data_vars, coords={"epoch": epoch})

    return grouped_data


def test_decom_packets(xarray_data, swe_test_data, fields_to_test):
    """This function checks that all instrument parameters are accounted for."""
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


def test_decompress_counts():
    """Test that we get correct decompressed counts from the algorithm."""
    expected_value = 24063
    input_count = 230
    returned_value = decompress_counts(np.array([input_count]))
    assert np.all(expected_value == returned_value)


def test_phi_to_bin():
    """Test phi_to_bin function."""

    # Define expected phi-to-bin mapping for one full spin
    phis = [
        12, 24, 36, 48, 60, 72, 84, 96, 108, 120,
        132, 144, 156, 168, 180, 192, 204, 216,
        228, 240, 252, 264, 276, 288, 300, 312,
        324, 336, 348, 360
    ]

    expected_bins = np.arange(30)

    for phi, expected_bin in zip(phis, expected_bins):
        assert phi_to_bin(phi) == expected_bin


def test_prepare_raw_counts():
    """Test that prepare_raw_counts correctly bins counts into (30, 7, 4) array."""

    # 2 rows = 4 phis (12, 24, 36, 48)
    epochs = [0, 1]

    data = {
        "group": ("epoch", [1, 1]),  # Both rows belong to group 1

        # CEM 1 (Phi 12, 24, 36, 48)
        "swe_cem1_e1": ("epoch", [1, 9]),
        "swe_cem1_e2": ("epoch", [2, 10]),
        "swe_cem1_e3": ("epoch", [3, 11]),
        "swe_cem1_e4": ("epoch", [4, 12]),

        # CEM 2
        "swe_cem2_e1": ("epoch", [5, 13]),
        "swe_cem2_e2": ("epoch", [6, 14]),
        "swe_cem2_e3": ("epoch", [7, 15]),
        "swe_cem2_e4": ("epoch", [8, 16]),
    }

    grouped_data = xr.Dataset(data, coords={"epoch": epochs})

    # Run the function
    raw_counts = prepare_raw_counts(grouped_data, group=1)

    # Expected shape (30, 7, 4) but only some phis are filled
    expected = np.zeros((30, 2, 4), dtype=np.uint8)

    # Fill expected values (matching phi bins for 12, 24, 36, 48)
    phi_bin_12 = 0  # Phi 12
    phi_bin_24 = 1  # Phi 24
    phi_bin_36 = 2  # Phi 36
    phi_bin_48 = 3  # Phi 48

    # CEM 1, Phi 12 (E1, E2)
    expected[phi_bin_12, 0, 0] = 1
    expected[phi_bin_12, 0, 1] = 2

    # CEM 1, Phi 24 (E3, E4)
    expected[phi_bin_24, 0, 2] = 3
    expected[phi_bin_24, 0, 3] = 4

    # CEM 1, Phi 36 (E1, E2)
    expected[phi_bin_36, 0, 0] = 9
    expected[phi_bin_36, 0, 1] = 10

    # CEM 1, Phi 48 (E3, E4)
    expected[phi_bin_48, 0, 2] = 11
    expected[phi_bin_48, 0, 3] = 12

    # CEM 2, Phi 12 (E1, E2)
    expected[phi_bin_12, 1, 0] = 5
    expected[phi_bin_12, 1, 1] = 6

    # CEM 2, Phi 24 (E3, E4)
    expected[phi_bin_24, 1, 2] = 7
    expected[phi_bin_24, 1, 3] = 8

    # CEM 2, Phi 36 (E1, E2)
    expected[phi_bin_36, 1, 0] = 13
    expected[phi_bin_36, 1, 1] = 14

    # CEM 2, Phi 48 (E3, E4)
    expected[phi_bin_48, 1, 2] = 15
    expected[phi_bin_48, 1, 3] = 16

    # Compare
    assert np.array_equal(raw_counts, expected)


@patch(
    "imap_processing.ialirt.l0.process_swe.read_in_flight_cal_data",
    return_value=pd.DataFrame(
        {
            "met_time": [453051300, 453051900],
            "cem1": [1, 2],
            "cem2": [1, 2],
            "cem3": [1, 2],
            "cem4": [1, 2],
            "cem5": [1, 2],
            "cem6": [1, 2],
            "cem7": [1, 2],
        }
    ),
)
def test_process_swe(mock_read_cal, swe_test_data, fields_to_test):
    """Test processing for swe."""
    swe_test_data = swe_test_data.rename(
        columns={v: k for k, v in fields_to_test.items()}
    )
    swe_test_data.index.name = "epoch"
    ds = swe_test_data.to_xarray()
    ds["src_seq_ctr"] = ("epoch", np.arange(len(ds["swe_shcoarse"])))
    swe_data = process_swe(ds)

    assert swe_data == []
