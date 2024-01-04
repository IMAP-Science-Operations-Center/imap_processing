import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets
from imap_processing.swapi.l1.swapi_l1 import (
    SWAPIAPID,
    check_for_bad_data,
    decompress_count,
    find_sweep_starts,
    get_indices_of_full_sweep,
    process_sweep_data,
)
from imap_processing.swapi.swapi_utils import create_dataset
from imap_processing.utils import group_by_apid, sort_by_time


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    test_folder_path = "swapi/tests/l0_data"
    packet_files = list(imap_module_directory.glob(f"{test_folder_path}/*.bin"))
    packet_definition = (
        f"{imap_module_directory}/swapi/packet_definitions/swapi_packet_definition.xml"
    )
    data_list = []
    for packet_file in packet_files:
        data_list.extend(decom_packets(packet_file, packet_definition))
    return data_list


def test_check_for_bad_data():
    """Test for bad data"""
    # create test data for this test
    total_sweeps = 3
    ds = xr.Dataset(
        {
            "PLAN_ID_SCIENCE": xr.DataArray(np.full((total_sweeps * 12), 1)),
            "SWEEP_TABLE": xr.DataArray(np.repeat(np.arange(total_sweeps), 12)),
            "MODE": xr.DataArray(np.full((total_sweeps * 12), 2)),
        }
    )

    # Check for no bad data
    bad_data_indices = check_for_bad_data(ds)
    assert len(bad_data_indices) == 0

    # Check for bad MODE data.
    # This test returns this indices because MODE has 0, 1 values
    # for the first two sweeps.
    # TODO: update test when we update MODE from HVENG to HVSCI
    ds["MODE"] = xr.DataArray(np.repeat(np.arange(total_sweeps), 12))
    bad_data_indices = check_for_bad_data(ds)
    assert np.all(bad_data_indices == np.arange(0, 24))

    # Check for bad SWEEP_TABLE data.
    # Reset MODE data and create first sweep to be mixed value
    ds["MODE"] = xr.DataArray(np.full((total_sweeps * 12), 2))
    ds["SWEEP_TABLE"][:12] = np.arange(0, 12)
    assert np.all(check_for_bad_data(ds) == np.arange(0, 12))

    # Check for bad PLAN_ID_SCIENCE data.
    ds["SWEEP_TABLE"] = xr.DataArray(np.repeat(np.arange(total_sweeps), 12))
    ds["PLAN_ID_SCIENCE"][24 : total_sweeps * 12] = np.arange(0, 12)
    assert np.all(check_for_bad_data(ds) == np.arange(24, total_sweeps * 12))


def test_decompress_count():
    """Test for decompress count"""
    # Test compressed and not overflow case
    raw_value = np.array([[12]])
    compression_flag = np.array([[1]])
    expected_value = raw_value[0][0] * 16
    returned_value = decompress_count(raw_value, compression_flag)
    assert returned_value[0][0] == expected_value

    # Test compressed and overflow case
    raw_value = np.array([[0xFFFF]])
    compression_flag = np.array([[1]])
    expected_value = -1
    returned_value = decompress_count(raw_value, compression_flag)
    assert returned_value[0][0] == expected_value

    # Test not compressed case
    raw_value = np.array([[12]])
    compression_flag = np.array([[0]])
    expected_value = 12
    returned_value = decompress_count(raw_value, compression_flag)
    assert returned_value[0][0] == expected_value


def test_find_sweep_starts():
    """Test for find sweep starts"""
    time = np.arange(26)
    sequence_number = time % 12
    ds = xr.Dataset({"SEQ_NUMBER": sequence_number}, coords={"Epoch": time})

    start_indices = find_sweep_starts(ds)
    assert np.all(start_indices == [0, 12])

    ds["SEQ_NUMBER"].data[:12] = np.arange(3, 15)
    start_indices = find_sweep_starts(ds)
    assert np.all(start_indices == [12])

    ds["SEQ_NUMBER"] = np.arange(3, 29)
    start_indices = find_sweep_starts(ds)
    assert np.all(start_indices == [])


def test_get_full_indices():
    """Test for correct full sweep indices"""
    time = np.arange(14)
    sequence_number = time % 12
    ds = xr.Dataset({"SEQ_NUMBER": sequence_number}, coords={"Epoch": time})

    sweep_indices = get_indices_of_full_sweep(ds)
    assert np.all(sweep_indices == np.arange(0, 12))


def test_swapi_algorithm(decom_test_data):
    """Test SWAPI L1 algorithm"""
    grouped_data = group_by_apid(decom_test_data)
    science_data = grouped_data[SWAPIAPID.SWP_SCI]
    sorted_packets = sort_by_time(science_data, "SHCOARSE")
    ds_data = create_dataset(sorted_packets)
    full_sweep_indices = get_indices_of_full_sweep(ds_data)
    full_sweep_sci = ds_data.isel({"Epoch": full_sweep_indices})
    total_packets = len(full_sweep_sci["SEQ_NUMBER"].data)
    # It takes 12 sequence data to make one full sweep
    total_sequence = 12
    total_full_sweeps = total_packets // total_sequence
    pcem_counts = process_sweep_data(full_sweep_sci, "PCEM_CNT", total_full_sweeps)
    # check that return value has correct shape
    assert pcem_counts.shape == (total_full_sweeps, 72)
    expected_count = [
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        31,
        30,
        29,
        28,
        27,
        26,
        25,
        24,
        23,
        22,
        21,
        20,
        19,
        18,
        17,
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
    ]

    assert np.all(pcem_counts == expected_count)
