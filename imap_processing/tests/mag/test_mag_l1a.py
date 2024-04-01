from pathlib import Path

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
    l0, _ = decom_packets(str(test_file))

    l1_mago, l1_magi = process_packets(l0)

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
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l1_test_data.pkts"
    l0 = decom_packets(str(test_file))

    mag_l0 = l0[0][0]

    # TODO rewrite this test with reverse-calculated unsigned 16 bit ints

    total_primary_vectors = (mag_l0.PUS_SSUBTYPE + 1) * mag_l0.PRI_VECSEC
    total_secondary_vectors = (mag_l0.PUS_SSUBTYPE + 1) * mag_l0.SEC_VECSEC

    test_vectors = mag_l0.VECTORS

    # 36 bytes
    (primary_vectors, secondary_vectors) = MagL1a.process_vector_data(
        test_vectors, total_primary_vectors, total_secondary_vectors
    )

    validation_data = pd.read_csv(current_directory / "mag_l1a_test_output.csv")

    for index in range(total_primary_vectors):
        assert primary_vectors[index][0] == validation_data.iloc[index]["x_pri"]
        assert primary_vectors[index][1] == validation_data.iloc[index]["y_pri"]
        assert primary_vectors[index][2] == validation_data.iloc[index]["z_pri"]
        assert primary_vectors[index][3] == validation_data.iloc[index]["rng_pri"]

        assert secondary_vectors[index][0] == validation_data.iloc[index]["x_sec"]
        assert secondary_vectors[index][1] == validation_data.iloc[index]["y_sec"]
        assert secondary_vectors[index][2] == validation_data.iloc[index]["z_sec"]
        assert secondary_vectors[index][3] == validation_data.iloc[index]["rng_sec"]


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
