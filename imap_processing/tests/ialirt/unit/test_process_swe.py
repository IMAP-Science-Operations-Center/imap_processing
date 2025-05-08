"""Tests to support I-ALiRT SWE packet parsing."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_swe import (
    average_counts,
    azimuthal_check_counterstreaming,
    compute_bidirectional,
    decompress_counts,
    determine_streaming,
    find_bin_offsets,
    find_min_counts,
    get_ialirt_energies,
    normalize_counts,
    phi_to_bin,
    polar_check_counterstreaming,
    prepare_raw_counts,
    process_swe,
)
from imap_processing.swe.utils.swe_constants import (
    ESA_VOLTAGE_ROW_INDEX_DICT,
    GEOMETRIC_FACTORS,
)
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
        / "data"
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
        / "data"
        / "l0"
        / "idle_export_eu.SWE_IALIRT_20240827_093852.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture
def xarray_data(binary_packet_path, xtce_swe_path):
    """Create xarray data"""
    apid = 1360

    xarray_data = packet_file_to_datasets(
        binary_packet_path, xtce_swe_path, use_derived_value=True
    )[apid]
    return xarray_data


@pytest.fixture
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


@pytest.fixture
def summed_half_cycle():
    """Create test set with known peaks"""

    summed_half_cycle = np.zeros((8, 30))

    for i in range(8):
        peak = i + 5
        summed_half_cycle[i, peak] = 100
        summed_half_cycle[i, (peak + 6) % 30] = 60  # +90 offset
        summed_half_cycle[i, (peak + 8) % 30] = 80  # +90 offset
        summed_half_cycle[i, (peak + 14) % 30] = 20  # +180 offset
        summed_half_cycle[i, (peak + 16) % 30] = 40  # +180 offset
        summed_half_cycle[i, (peak - 6) % 30] = 5  # -90 offset
        summed_half_cycle[i, (peak - 8) % 30] = 15  # -90 offset

    return summed_half_cycle


def test_get_energy():
    """Tests get_alirt_energies function."""
    energies = get_ialirt_energies()

    for i in range(len(energies)):
        assert i + 11 == ESA_VOLTAGE_ROW_INDEX_DICT[energies[i]]


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
    phis = np.arange(12, 361, 12).tolist()

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
        "swe_seq": ("epoch", [0, 1]),
    }

    grouped_data = xr.Dataset(data, coords={"epoch": epochs})
    group_mask = grouped_data["group"] == 1
    grouped = grouped_data.sel(epoch=group_mask)

    raw_counts = prepare_raw_counts(grouped, cem_number=2)

    # Expected shape (8, 7, 30) but only some CEMs are used.
    expected = np.zeros((8, 2, 30), dtype=np.uint8)

    # Phi bins for 12, 24, 36, 48)
    phi_bin_12 = 0  # Phi 12
    phi_bin_24 = 1  # Phi 24
    phi_bin_36 = 2  # Phi 36
    phi_bin_48 = 3  # Phi 48

    # CEM 1, Phi 12 (E1, E2)
    expected[1, 0, phi_bin_12] = 1
    expected[5, 0, phi_bin_12] = 2

    # CEM 1, Phi 24 (E3, E4)
    expected[7, 0, phi_bin_24] = 3
    expected[3, 0, phi_bin_24] = 4

    # CEM 1, Phi 36 (E1, E2)
    expected[1, 0, phi_bin_36] = 9
    expected[5, 0, phi_bin_36] = 10

    # CEM 1, Phi 48 (E3, E4)
    expected[7, 0, phi_bin_48] = 11
    expected[3, 0, phi_bin_48] = 12

    # CEM 2, Phi 12 (E1, E2)
    expected[1, 1, phi_bin_12] = 5
    expected[5, 1, phi_bin_12] = 6

    # CEM 2, Phi 24 (E3, E4)
    expected[7, 1, phi_bin_24] = 7
    expected[3, 1, phi_bin_24] = 8

    # CEM 2, Phi 36 (E1, E2)
    expected[1, 1, phi_bin_36] = 13
    expected[5, 1, phi_bin_36] = 14

    # CEM 2, Phi 48 (E3, E4)
    expected[7, 1, phi_bin_48] = 15
    expected[3, 1, phi_bin_48] = 16

    assert np.array_equal(raw_counts, expected)


def test_normalize_counts():
    """Tests normalize_counts function"""

    # Shape (2, 7, 3) for a small test case
    corrected_counts = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
            ],
            [
                [2, 4, 6],
                [8, 10, 12],
                [14, 16, 18],
                [20, 22, 24],
                [26, 28, 30],
                [32, 34, 36],
                [38, 40, 42],
            ],
        ],
        dtype=np.uint8,
    )

    latest_cal = pd.Series(
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        index=[
            "cem1",
            "cem2",
            "cem3",
            "cem4",
            "cem5",
            "cem6",
            "cem7",
        ],  # Simulating real data structure
        dtype=np.float64,
    )
    expected = np.zeros((2, 7, 3), dtype=np.float64)

    for i in range(2):
        for j in range(7):
            for k in range(3):
                if corrected_counts[i][j][k] < 0:
                    expected[i][j][k] = 0.0
                else:
                    expected[i][j][k] = (
                        corrected_counts[i][j][k] * latest_cal[j] / GEOMETRIC_FACTORS[j]
                    )

    norm_counts = normalize_counts(corrected_counts, latest_cal)

    assert np.allclose(norm_counts, expected, atol=1e-9)


def test_find_bin_offsets():
    """Tests find_bin_offsets function"""

    peak_bins = np.array([5, 6, 29])
    bins = find_bin_offsets(peak_bins, (2, 3))
    np.testing.assert_array_equal(bins, np.array([[7, 8, 1], [8, 9, 2]]))


def test_average_counts(summed_half_cycle):
    """Tests average_values_and_azimuth function"""

    # Find the azimuth angle that corresponds to the maximum counts at each energy
    peak_az_bin = np.argmax(summed_half_cycle, axis=1)

    # Bins +6 and +8 correspond to +90 degrees.
    counts_90 = average_counts(peak_az_bin, summed_half_cycle, (6, 8))

    assert np.allclose(counts_90, np.full(8, 70), atol=1e-9)


def test_find_min_counts(summed_half_cycle):
    """Tests find_min_counts function"""

    cpeak, cmin, counts = find_min_counts(summed_half_cycle)
    np.testing.assert_array_equal(cpeak, np.full(8, 100))
    np.testing.assert_array_equal(cmin, counts[0])


def test_determine_streaming_summed_cems():
    """Tests determine_streaming_summed_cems function."""

    # Case where streaming should be True
    cpeak = np.array([100, 80, 50, 60])
    cmin = np.array([20, 70, 40, 60])
    counts_180 = np.array([40, 60, 90, 110])
    assert np.array_equal(
        determine_streaming(cpeak, counts_180, cmin), np.array([1, 0, 0, 0])
    )


def test_compute_bidirectional():
    """Tests compute_bidirectional function."""

    first_half = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    second_half = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    assert compute_bidirectional(first_half, second_half) == (0, 0)

    first_half = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    second_half = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert compute_bidirectional(first_half, second_half) == (1, 0)


def test_azimuthal_check_counterstreaming(summed_half_cycle):
    """Tests azimuthal_check_counterstreaming function."""

    bde = azimuthal_check_counterstreaming(summed_half_cycle, summed_half_cycle)

    assert bde == (1, 1)


def test_polar_check_counterstreaming():
    """Tests polar_check_counterstreaming function."""

    # cem0 (cem1) and cem6 (cem7) have high values
    # cem2, cem3, cem4 (cem3 to cem5) are low and used for cmin
    row = np.array([100, 20, 5, 5, 5, 20, 100])
    summed_half = np.tile(row, (8, 1))

    bde = polar_check_counterstreaming(summed_half, summed_half)

    assert bde == (1, 1)


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
    in_flight_cal_file = (
        imap_module_directory
        / "tests/swe/lut/imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"
    )
    swe_data = process_swe(ds, [in_flight_cal_file])

    # TODO: add tests with test data here.

    # Check that all groups in the data are accounted for.
    assert len(swe_data) == 912 // 60
