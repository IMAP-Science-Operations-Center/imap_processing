import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.decom import decom_packets
from imap_processing.swapi.l1.swapi_l1 import (
    SWAPIAPID,
    decompress_count,
    filter_good_data,
    find_sweep_starts,
    get_indices_of_full_sweep,
    process_swapi_science,
    process_sweep_data,
    swapi_l1,
)
from imap_processing.utils import create_dataset, group_by_apid, sort_by_time


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    test_folder_path = "tests/swapi/l0_data"
    packet_files = list(imap_module_directory.glob(f"{test_folder_path}/*.bin"))
    packet_definition = (
        f"{imap_module_directory}/swapi/packet_definitions/swapi_packet_definition.xml"
    )
    data_list = []
    for packet_file in packet_files:
        data_list.extend(decom_packets(packet_file, packet_definition))
    return data_list


def test_filter_good_data():
    """Test for bad data"""
    # create test data for this test
    total_sweeps = 3
    ds = xr.Dataset(
        {
            "plan_id_science": xr.DataArray(np.full((total_sweeps * 12), 1)),
            "sweep_table": xr.DataArray(np.repeat(np.arange(total_sweeps), 12)),
            "mode": xr.DataArray(np.full((total_sweeps * 12), 2)),
        },
        coords={"epoch": np.arange(total_sweeps * 12)},
    )

    # Check for no bad data
    bad_data_indices = filter_good_data(ds)
    assert len(bad_data_indices) == 36

    # Check for bad MODE data.
    # This test returns this indices because MODE has 0, 1 values
    # for the first two sweeps.
    # TODO: update test when we update MODE from HVENG to HVSCI
    ds["mode"] = xr.DataArray(np.repeat(np.arange(total_sweeps), 12))
    bad_data_indices = filter_good_data(ds)
    np.testing.assert_array_equal(bad_data_indices, np.arange(24, 36))

    # Check for bad sweep_table data.
    # Reset MODE data and create first sweep to be mixed value
    ds["mode"] = xr.DataArray(np.full((total_sweeps * 12), 2))
    ds["sweep_table"][:12] = np.arange(0, 12)
    np.testing.assert_array_equal(filter_good_data(ds), np.arange(12, 36))

    ds["sweep_table"][24:] = np.arange(0, 12)
    expected = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    np.testing.assert_array_equal(filter_good_data(ds), expected)

    # Check for bad plan_id_science data.
    ds["sweep_table"] = xr.DataArray(np.repeat(np.arange(total_sweeps), 12))
    ds["plan_id_science"][24 : total_sweeps * 12] = np.arange(0, 12)
    np.testing.assert_array_equal(filter_good_data(ds), np.arange(0, 24))


def test_decompress_count():
    """Test for decompress count"""
    # compressed + no-overflow, compressed + overflow, no compression
    raw_values = np.array([[12, 0xFFFF, 12]])
    compression_flag = np.array([[1, 1, 0]])
    expected = np.array([[12 * 16, -1, 12]])
    returned_value = decompress_count(raw_values, compression_flag)
    np.testing.assert_array_equal(returned_value, expected)


def test_find_sweep_starts():
    """Test for find sweep starts"""
    time = np.arange(26)
    sequence_number = time % 12
    ds = xr.Dataset({"seq_number": sequence_number}, coords={"epoch": time})

    start_indices = find_sweep_starts(ds)
    np.testing.assert_array_equal(start_indices, [0, 12])

    ds["seq_number"].data[:12] = np.arange(3, 15)
    start_indices = find_sweep_starts(ds)
    np.testing.assert_array_equal(start_indices, [12])

    # Creating test data that doesn't have start sequence.
    # Sequence number range is 0-11.
    ds["seq_number"] = np.arange(3, 29)
    start_indices = find_sweep_starts(ds)
    np.testing.assert_array_equal(start_indices, [])


def test_get_full_indices():
    """Test for correct full sweep indices"""
    time = np.arange(26)
    sequence_number = time % 12
    ds = xr.Dataset({"seq_number": sequence_number}, coords={"epoch": time})

    sweep_indices = get_indices_of_full_sweep(ds)
    np.testing.assert_array_equal(sweep_indices, np.arange(0, 24))


def test_swapi_algorithm(decom_test_data):
    """Test SWAPI L1 algorithm"""
    grouped_data = group_by_apid(decom_test_data)
    science_data = grouped_data[SWAPIAPID.SWP_SCI]
    sorted_packets = sort_by_time(science_data, "SHCOARSE")
    ds_data = create_dataset(sorted_packets, include_header=False)
    full_sweep_indices = get_indices_of_full_sweep(ds_data)
    full_sweep_sci = ds_data.isel({"epoch": full_sweep_indices})
    total_packets = len(full_sweep_sci["seq_number"].data)
    # It takes 12 sequence data to make one full sweep
    total_sequence = 12
    total_full_sweeps = total_packets // total_sequence
    pcem_counts = process_sweep_data(full_sweep_sci, "pcem_cnt")
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

    np.testing.assert_array_equal(pcem_counts[0], expected_count)


def test_process_swapi_science(decom_test_data):
    """Test process swapi science"""
    grouped_data = group_by_apid(decom_test_data)
    science_data = grouped_data[SWAPIAPID.SWP_SCI]
    sorted_packets = sort_by_time(science_data, "SHCOARSE")
    ds_data = create_dataset(sorted_packets, include_header=False)
    processed_data = process_swapi_science(ds_data)

    # Test dataset dimensions
    assert processed_data.sizes == {"epoch": 3, "energy": 72}
    # Test epoch data is correct
    expected_epoch_datetime = [
        np.datetime64("2010-01-01T00:01:54.184000000"),
        np.datetime64("2010-01-01T00:02:06.184000000"),
        np.datetime64("2010-01-01T00:02:18.184000000"),
    ]
    np.testing.assert_array_equal(processed_data["epoch"].data, expected_epoch_datetime)

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
    # Test that we got expected counts and datashape
    np.testing.assert_array_equal(processed_data["swp_pcem_counts"][0], expected_count)
    assert processed_data["swp_pcem_counts"].shape == (3, 72)
    # Test that we calculated uncertainty correctly
    np.testing.assert_allclose(
        np.sqrt(processed_data["swp_pcem_counts"][0]), processed_data["swp_pcem_err"][0]
    )

    # make PLAN_ID data incorrect
    ds_data["plan_id_science"][:12] = np.arange(12)
    processed_data = process_swapi_science(ds_data)

    # Test dataset dimensions
    assert processed_data.sizes == {"epoch": 2, "energy": 72}

    # Test CDF File
    # This time mismatch is because of sample data. Sample data has
    # SHCOARSE time as 48, 60, 72. That's why time is different.
    cdf_filename = "imap_swapi_l1_sci-1min_20100101_v001.cdf"
    cdf_path = write_cdf(processed_data)
    assert cdf_path.name == cdf_filename


def test_swapi_l1_hk(decom_test_data):
    """Test housekeeping processing and CDF file creation"""
    grouped_data = group_by_apid(decom_test_data)
    processed_data = swapi_l1(grouped_data[SWAPIAPID.SWP_HK])

    # Test CDF File
    cdf_filename = "imap_swapi_l1_hk_20100101_v001.cdf"
    cdf_path = write_cdf(processed_data[0])

    assert cdf_path.name == cdf_filename
