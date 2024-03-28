from pathlib import Path

import pandas as pd

from imap_processing.mag.l0.decom_mag import decom_packets
from imap_processing.mag.l1a.mag_l1a import process_vector_data
from imap_processing.mag.l1a.mag_l1a_data import MAX_FINE_TIME, TimeTuple


def test_process_vector_data():
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l1_test_data.pkts"
    l0 = decom_packets(str(test_file))
    mag_l0 = l0[0]

    total_primary_vectors = (mag_l0.PUS_SSUBTYPE + 1) * mag_l0.PRI_VECSEC
    total_secondary_vectors = (mag_l0.PUS_SSUBTYPE + 1) * mag_l0.SEC_VECSEC

    test_vectors = l0[0].VECTORS

    # 36 bytes
    (primary_vectors, secondary_vectors) = process_vector_data(
        test_vectors, total_primary_vectors, total_secondary_vectors
    )

    validation_data = pd.read_csv(current_directory / "mag_l1a_test_output.csv")

    print(validation_data.index)
    for index in range(total_primary_vectors):
        # print(validation_data.iloc[index])
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
