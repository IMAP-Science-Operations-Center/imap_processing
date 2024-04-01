from pathlib import Path

import numpy as np
import pandas as pd

from imap_processing.mag.l0.decom_mag import decom_packets
from imap_processing.mag.l1a.mag_l1a import process_packets
from imap_processing.mag.l1a.mag_l1a_data import (
    MAX_FINE_TIME,
    MagL1a,
    TimeTuple,
    Vector,
)


def test_compare_validation_data():
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l1_test_data.pkts"
    # Test file contains only normal packets
    l0 = decom_packets(str(test_file))

    l1 = process_packets(l0["norm"])
    l1_mago = l1["mago"]
    l1_magi = l1["magi"]

    assert len(l1_mago) == 6
    assert len(l1_magi) == 6

    validation_data = pd.read_csv(current_directory / "mag_l1a_test_output.csv")

    vector_index = 0

    # Validation data does not have differing timestamps
    for index in validation_data.index:
        # Sequence in validation data starts at 5
        # Mago is primary, Magi is secondary in test data
        l1_pri = l1_mago[validation_data["sequence"][index] - 5]
        l1_sec = l1_magi[validation_data["sequence"][index] - 5]

        assert l1_pri.vectors[vector_index].x == validation_data["x_pri"][index]
        assert l1_pri.vectors[vector_index].y == validation_data["y_pri"][index]
        assert l1_pri.vectors[vector_index].z == validation_data["z_pri"][index]
        assert l1_pri.vectors[vector_index].rng == validation_data["rng_pri"][index]

        assert l1_sec.vectors[vector_index].x == validation_data["x_sec"][index]
        assert l1_sec.vectors[vector_index].y == validation_data["y_sec"][index]
        assert l1_sec.vectors[vector_index].z == validation_data["z_sec"][index]
        assert l1_sec.vectors[vector_index].rng == validation_data["rng_sec"][index]

        vector_index = (
            0 if vector_index == l1_pri.expected_vector_count - 1 else vector_index + 1
        )


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
    test_time_tuple = TimeTuple(439067318, 64618)

    test_add = test_time_tuple + 2

    assert test_add == TimeTuple(439067320, 64618)

    # 1 / MAX_FINE_TIME
    test_add = test_time_tuple + 1 / MAX_FINE_TIME

    assert test_add == TimeTuple(439067318, 64619)

    test_add = test_time_tuple + (1000 / MAX_FINE_TIME)

    assert test_add == TimeTuple(439067319, 83)


def test_vector_time():
    test_data = MagL1a(
        True,
        True,
        TimeTuple(10000, 0),
        2,  # 2 vectors per second
        4,
        2,
        0,
        [(1, 2, 3, 4), (1, 2, 3, 4), (2, 2, 2, 3), (3, 3, 3, 4)],
    )

    assert test_data.vectors[0] == Vector((1, 2, 3, 4), TimeTuple(10000, 0))
    assert test_data.vectors[1] == Vector((1, 2, 3, 4), TimeTuple(10000, 32768))
    assert test_data.vectors[2] == Vector((2, 2, 2, 3), TimeTuple(10001, 1))
    assert test_data.vectors[3] == Vector((3, 3, 3, 4), TimeTuple(10001, 32769))
