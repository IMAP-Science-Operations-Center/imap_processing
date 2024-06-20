from pathlib import Path

import numpy as np
import pandas as pd

from imap_processing.cdf.utils import convert_met_to_datetime64
from imap_processing.mag.l0.decom_mag import decom_packets
from imap_processing.mag.l1a.mag_l1a import process_packets
from imap_processing.mag.l1a.mag_l1a_data import (
    MAX_FINE_TIME,
    MagL1a,
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

    converted_start_time_ns = convert_met_to_datetime64(start_time.to_seconds())

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
