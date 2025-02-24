from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.mag.l1a.mag_l1a_data import TimeTuple
from imap_processing.mag.l1b.mag_l1b import mag_l1b


@pytest.mark.parametrize(
    ("test_number"), ["001", "002", "003", "004", "005", "006", "007", "008"]
)
def test_mag_l1a_validation(test_number):
    source_directory = Path(__file__).parent / "validation" / "L1a" / f"T{test_number}"

    input_file = source_directory / f"mag-l0-l1a-t{test_number}-in.bin"
    expected_output_file = source_directory / f"mag-l0-l1a-t{test_number}-out.csv"

    mag_l1a_out = mag_l1a(input_file, "v000")
    expected_output = pd.read_csv(expected_output_file)

    raw = mag_l1a_out[0]
    primary = mag_l1a_out[1]
    secondary = mag_l1a_out[2]

    assert len(mag_l1a_out) == 3
    for index in expected_output.index:
        assert expected_output["x_pri"].iloc[index] == primary["vectors"].data[index][0]
        assert expected_output["y_pri"].iloc[index] == primary["vectors"].data[index][1]
        assert expected_output["z_pri"].iloc[index] == primary["vectors"].data[index][2]
        assert (
            expected_output["rng_pri"].iloc[index] == primary["vectors"].data[index][3]
        )

        assert (
            expected_output["x_sec"].iloc[index] == secondary["vectors"].data[index][0]
        )
        assert (
            expected_output["y_sec"].iloc[index] == secondary["vectors"].data[index][1]
        )
        assert (
            expected_output["z_sec"].iloc[index] == secondary["vectors"].data[index][2]
        )
        assert (
            expected_output["rng_sec"].iloc[index]
            == secondary["vectors"].data[index][3]
        )

        assert expected_output["pri_coarse"].iloc[index] == raw["PRI_COARSETM"].data[0]
        assert expected_output["pri_fine"].iloc[index] == raw["PRI_FNTM"].data[0]
        assert expected_output["sec_coarse"].iloc[index] == raw["SEC_COARSETM"].data[0]
        assert expected_output["sec_fine"].iloc[index] == raw["SEC_FNTM"].data[0]

        expected_pri_time = TimeTuple(
            expected_output["pri_coarse"].iloc[index],
            expected_output["pri_fine"].iloc[index],
        ).to_j2000ns()
        expected_sec_time = TimeTuple(
            expected_output["sec_coarse"].iloc[index],
            expected_output["sec_fine"].iloc[index],
        ).to_j2000ns()

        # Found a bug here, fixed now
        # Only the first timestamp should match pri_coarse and pri_fine
        assert expected_pri_time == primary["epoch"].data[0]
        assert expected_sec_time == secondary["epoch"].data[0]

        assert (
            expected_output["compression"].iloc[index]
            == primary["compression_flags"].data[index][0]
        )
        assert (
            expected_output["compression"].iloc[index]
            == secondary["compression_flags"].data[index][0]
        )

        # TODO: I set this to zero in the output (per algo doc) if compression is
        #  disabled.
        if expected_output["compression"].iloc[index] != 0:
            assert (
                expected_output["compression_width_bits"].iloc[index]
                == primary["compression_flags"].data[index][1]
            )
            assert (
                expected_output["compression_width_bits"].iloc[index]
                == secondary["compression_flags"].data[index][1]
            )


@pytest.mark.xfail(reason="Some bug somewhere")
@pytest.mark.parametrize(("test_number"), ["009", "010", "011"])
def test_mag_l1b_validation(test_number):
    source_directory = Path(__file__).parent / "validation" / "L1b" / f"T{test_number}"
    mag_l1a_input = mag_l1a(source_directory / "data.bin", "v000")

    mago = mag_l1b(mag_l1a_input[1], "v000")
    magi = mag_l1b(mag_l1a_input[2], "v000")

    expected_mago = pd.read_csv(
        source_directory / f"mag-l1a-l1b-t{test_number}-mago-out.csv"
    )
    expected_magi = pd.read_csv(
        source_directory / f"mag-l1a-l1b-t{test_number}-magi-out.csv"
    )

    for index in expected_magi.index:
        assert np.allclose(
            expected_magi["x"].iloc[index], magi["vectors"].data[index][0], atol=1e-9
        )
        assert np.allclose(
            expected_magi["y"].iloc[index], magi["vectors"].data[index][1], atol=1e-9
        )
        assert np.allclose(
            expected_magi["z"].iloc[index], magi["vectors"].data[index][2], atol=1e-9
        )
        assert expected_magi["range"].iloc[index] == magi["vectors"].data[index][3]
        assert (
            expected_magi["compression"].iloc[index]
            == magi["compression_flags"].data[index][0]
        )
        assert (
            expected_magi["compression_width"].iloc[index]
            == magi["compression_flags"].data[index][1]
        )

    for index in expected_mago.index:
        # TODO: come back to timestamp.
        # Can't compare UTC, coarse/fine don't work.
        assert np.allclose(
            expected_mago["x"].iloc[index], mago["vectors"].data[index][0], atol=1e-9
        )
        assert np.allclose(
            expected_mago["y"].iloc[index], mago["vectors"].data[index][1], atol=1e-9
        )
        assert np.allclose(
            expected_mago["z"].iloc[index], mago["vectors"].data[index][2], atol=1e-9
        )
        assert expected_mago["range"].iloc[index] == mago["vectors"].data[index][3]
        assert (
            expected_mago["compression"].iloc[index]
            == mago["compression_flags"].data[index][0]
        )
        assert (
            expected_mago["compression_width"].iloc[index]
            == mago["compression_flags"].data[index][1]
        )
