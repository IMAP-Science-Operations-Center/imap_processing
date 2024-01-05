import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.swe.l0 import decom_swe
from imap_processing.swe.l1a.swe_science import decompressed_counts, swe_science


@pytest.fixture(scope="session")
def decom_test_data():
    """Read test data from file"""
    # NOTE: data was provided in this sequence in both bin and validation data
    # from instrument team.
    # Packet 1 has spin 4's data
    # Packet 2 has spin 1's data
    # Packet 3 has spin 2's data
    # Packet 4 has spin 3's data
    # reorder to match the order of spin in the data
    packet_files = [
        imap_module_directory
        / "tests/swe/l0_data/20230927173253_SWE_SCIENCE_packet.bin",
        imap_module_directory
        / "tests/swe/l0_data/20230927173308_SWE_SCIENCE_packet.bin",
        imap_module_directory
        / "tests/swe/l0_data/20230927173323_SWE_SCIENCE_packet.bin",
        imap_module_directory
        / "tests/swe/l0_data/20230927173238_SWE_SCIENCE_packet.bin",
    ]
    data_list = []
    for packet_file in packet_files:
        data_list.extend(decom_swe.decom_packets(packet_file))
    return data_list


def test_number_of_packets(decom_test_data):
    """This test and validate number of packets."""
    expected_number_of_packets = 4
    assert len(decom_test_data) == expected_number_of_packets


def test_decompress_algorithm():
    """Test that we get correct decompressed counts from the algorithm."""
    expected_value = 24063
    input_count = 230
    returned_value = decompressed_counts(input_count)
    assert expected_value == returned_value


def test_swe_raw_science_data(decom_test_data):
    """This test and validate raw and derived data of SWE science data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    raw_validation_data = pd.read_csv(
        test_data_path / "idle_export_raw.SWE_SCIENCE_20230927_172708.csv",
        index_col="SHCOARSE",
    )

    first_data = decom_test_data[0]
    validation_data = raw_validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    # compare raw values of housekeeping data
    for key, value in first_data.data.items():
        if key == "SHCOARSE":
            # compare SHCOARSE value
            assert value.raw_value == validation_data.name
            continue
        if key == "SCIENCE_DATA":
            continue
        # check if the data is the same
        assert value.raw_value == validation_data[key]


def test_swe_derived_science_data(decom_test_data):
    """This test and validate raw and derived data of SWE science data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    derived_validation_data = pd.read_csv(
        test_data_path / "idle_export_eu.SWE_SCIENCE_20230927_172708.csv",
        index_col="SHCOARSE",
    )

    first_data = decom_test_data[0]
    validation_data = derived_validation_data.loc[first_data.data["SHCOARSE"].raw_value]

    enum_name_list = [
        "CEM_NOMINAL_ONLY",
        "SPIN_PERIOD_VALIDITY",
        "SPIN_PHASE_VALIDITY",
        "SPIN_PERIOD_SOURCE",
        "REPOINT_WARNING",
        "HIGH_COUNT",
        "STIM_ENABLED",
        "QUARTER_CYCLE",
    ]
    # check ENUM values
    for enum_name in enum_name_list:
        assert first_data.data[enum_name].derived_value == validation_data[enum_name]


def test_data_order(decom_test_data):
    # test that the data is in right order
    assert decom_test_data[0].data["QUARTER_CYCLE"].derived_value == "FIRST"
    assert decom_test_data[1].data["QUARTER_CYCLE"].derived_value == "SECOND"
    assert decom_test_data[2].data["QUARTER_CYCLE"].derived_value == "THIRD"
    assert decom_test_data[3].data["QUARTER_CYCLE"].derived_value == "FORTH"

    # Get unpacked science data
    processed_data = swe_science(decom_test_data)

    quarter_cycle = processed_data["QUARTER_CYCLE"].data
    assert quarter_cycle[0] == 0
    assert quarter_cycle[1] == 1
    assert quarter_cycle[2] == 2
    assert quarter_cycle[3] == 3


def test_swe_science_algorithm(decom_test_data):
    """Test general shape of return dataset from swe_science."""
    # Get unpacked science data
    processed_data = swe_science(decom_test_data)

    # science data should have this shape, 15x12x7.
    science_data = processed_data["SCIENCE_DATA"].data[0]
    assert science_data.shape == (180, 7)

    # Test data has n packets, therefore, SPIN_PHASE should have that same length.
    spin_phase = processed_data["SPIN_PHASE"]
    expected_length = 4
    assert len(spin_phase) == expected_length


def test_decompress_counts(decom_test_data):
    """Test decompress counts."""
    test_data_path = imap_module_directory / "tests/swe/decompressed"
    filepaths = [
        "20230927173253_1st_quarter_decompressed.csv",
        "20230927173308_2nd_quarter_decompressed.csv",
        "20230927173323_3rd_quarter_decompressed.csv",
        "20230927173238_4th_quarter_decompressed.csv",
    ]
    decompressed_data = swe_science(decom_test_data)

    for index in range(len(filepaths)):
        instrument_decompressed_counts = pd.read_csv(
            test_data_path / f"{filepaths[index]}", index_col="Index"
        )

        assert (
            decompressed_data["QUARTER_CYCLE"].data[index]
            == decom_test_data[index].data["QUARTER_CYCLE"].raw_value
        )
        sdc_decompressed_counts = (
            decompressed_data["SCIENCE_DATA"].data[index].reshape(180, 7)
        )

        for i in range(7):
            cem_decompressed_counts = instrument_decompressed_counts[
                f"CEM {i+1}"
            ].values
            assert np.all(sdc_decompressed_counts[:, i] == cem_decompressed_counts)
