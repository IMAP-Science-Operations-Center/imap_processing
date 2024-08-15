from pathlib import Path

import numpy as np
import pandas as pd

from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.mag.l0.decom_mag import decom_packets
from imap_processing.mag.l1a.mag_l1a import mag_l1a, process_packets
from imap_processing.mag.l1a.mag_l1a_data import (
    MAX_FINE_TIME,
    MagL1a,
    MagL1aPacketProperties,
    TimeTuple,
)


def test_compare_validation_data():
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l1_test_data.pkts"
    # Test file contains only normal packets
    l0 = decom_packets(test_file)

    l1 = process_packets(l0["norm"])
    # Should have one day of data
    expected_day = np.datetime64("2023-11-30")
    l1_mago = l1["mago"][expected_day]
    l1_magi = l1["magi"][expected_day]

    assert len(l1_mago.vectors) == 96
    assert len(l1_magi.vectors) == 96

    validation_data = pd.read_csv(current_directory / "mag_l1a_test_output.csv")

    # Validation data does not have differing timestamps
    for index in validation_data.index:
        # Sequence in validation data starts at 5
        # Mago is primary, Magi is secondary in test data

        assert l1_mago.vectors[index][0] == validation_data["x_pri"][index]
        assert l1_mago.vectors[index][1] == validation_data["y_pri"][index]
        assert l1_mago.vectors[index][2] == validation_data["z_pri"][index]
        assert l1_mago.vectors[index][3] == validation_data["rng_pri"][index]

        assert l1_magi.vectors[index][0] == validation_data["x_sec"][index]
        assert l1_magi.vectors[index][1] == validation_data["y_sec"][index]
        assert l1_magi.vectors[index][2] == validation_data["z_sec"][index]
        assert l1_magi.vectors[index][3] == validation_data["rng_sec"][index]


def test_process_vector_data():
    expected_vector_data = [[1001, 1002, -3001, 3], [2001, -2002, -3333, 1]]

    # 100 bits, created by hand by appending all bits from expected_vector_data into one
    # hex string, with range being 2 bits (so the second half is offset from the hex
    # values)
    hex_string = "03E903EAF447C1F47E0BBCBED0"
    input_data = np.frombuffer(bytes.fromhex(hex_string), dtype=np.dtype(">b"))

    total_primary_vectors = 1
    total_secondary_vectors = 1

    # 36 bytes
    (primary_vectors, secondary_vectors) = MagL1a.process_vector_data(
        input_data, total_primary_vectors, total_secondary_vectors
    )

    assert primary_vectors[0][0] == expected_vector_data[0][0]
    assert primary_vectors[0][1] == expected_vector_data[0][1]
    assert primary_vectors[0][2] == expected_vector_data[0][2]
    assert primary_vectors[0][3] == expected_vector_data[0][3]

    assert secondary_vectors[0][0] == expected_vector_data[1][0]
    assert secondary_vectors[0][1] == expected_vector_data[1][1]
    assert secondary_vectors[0][2] == expected_vector_data[1][2]
    assert secondary_vectors[0][3] == expected_vector_data[1][3]


def test_time_tuple():
    example_time_tuple = TimeTuple(439067318, 64618)

    test_add = example_time_tuple + 2

    assert test_add == TimeTuple(439067320, 64618)

    # 1 / MAX_FINE_TIME
    test_add = example_time_tuple + 1 / MAX_FINE_TIME

    assert test_add == TimeTuple(439067318, 64619)

    test_add = example_time_tuple + (1000 / MAX_FINE_TIME)

    assert test_add == TimeTuple(439067319, 83)


def test_calculate_vector_time():
    test_vectors = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint
    )
    test_vecsec = 2
    start_time = TimeTuple(10000, 0)

    test_data = MagL1a.calculate_vector_time(test_vectors, test_vecsec, start_time)

    converted_start_time_ns = met_to_j2000ns(start_time.to_seconds())

    skips_ns = np.timedelta64(int(1 / test_vecsec * 1e9), "ns")
    expected_data = np.array(
        [
            [1, 2, 3, int(converted_start_time_ns)],
            [4, 5, 6, int(converted_start_time_ns + skips_ns)],
            [7, 8, 9, int(converted_start_time_ns + skips_ns * 2)],
            [10, 11, 12, int(converted_start_time_ns + skips_ns * 3)],
        ]
    )
    assert (test_data == expected_data).all()


def test_mag_l1a_data():
    test_vectors = np.array(
        [
            [1, 2, 3, 4, 10000],
            [5, 6, 7, 8, 10050],
            [9, 10, 11, 12, 10100],
            [13, 13, 11, 12, 10150],
        ],
        dtype=np.uint,
    )
    test_vecsec = 2
    start_time = TimeTuple(10000, 0)

    packet_properties = MagL1aPacketProperties(
        435954628, start_time, test_vecsec, 1, 0, 0, 1
    )
    mag_l1a = MagL1a(True, True, 10000, test_vectors, packet_properties)

    new_vectors = np.array(
        [[13, 14, 15, 16, 10400], [16, 17, 18, 19, 10450]], dtype=np.uint
    )

    new_seq = 5
    new_properties = MagL1aPacketProperties(
        435954628, TimeTuple(10400, 0), test_vecsec, 0, new_seq, 0, 1
    )
    mag_l1a.append_vectors(new_vectors, new_properties)

    assert np.array_equal(
        mag_l1a.vectors,
        np.array(
            [
                [1, 2, 3, 4, 10000],
                [5, 6, 7, 8, 10050],
                [9, 10, 11, 12, 10100],
                [13, 13, 11, 12, 10150],
                [13, 14, 15, 16, 10400],
                [16, 17, 18, 19, 10450],
            ],
            dtype=np.uint,
        ),
    )
    assert mag_l1a.missing_sequences == [1, 2, 3, 4]


def test_mag_l1a():
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l1_test_data.pkts"

    output_data = mag_l1a(test_file, "v001")

    # Test data is one day's worth of NORM data, so it should return one raw, one MAGO
    # and one MAGI dataset
    assert len(output_data) == 3
    expected_logical_source = [
        "imap_mag_l1a_norm-raw",
        "imap_mag_l1a_norm-mago",
        "imap_mag_l1a_norm-magi",
    ]

    for data_type in [data.attrs["Logical_source"] for data in output_data]:
        assert data_type in expected_logical_source

    for data in output_data:
        assert data.attrs["Data_version"] == "v001"
