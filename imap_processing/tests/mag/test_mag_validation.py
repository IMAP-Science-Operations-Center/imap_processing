from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.mag.l1a.mag_l1a_data import MagL1a, TimeTuple
from imap_processing.mag.l1b.mag_l1b import mag_l1b
from imap_processing.spice.time import str_to_et, ttj2000ns_to_et
from imap_processing.tests.mag.conftest import mag_l1a_dataset_generator


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

        assert expected_output["pri_coarse"].iloc[index] == raw["pri_coarsetm"].data[0]
        assert expected_output["pri_fine"].iloc[index] == raw["pri_fntm"].data[0]
        assert expected_output["sec_coarse"].iloc[index] == raw["sec_coarsetm"].data[0]
        assert expected_output["sec_fine"].iloc[index] == raw["sec_fntm"].data[0]

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


@pytest.mark.parametrize(("test_number"), ["009", "010", "011", "012"])
def test_mag_l1b_validation(test_number):
    source_directory = Path(__file__).parent / "validation" / "L1b" / f"T{test_number}"
    cdf_file = source_directory / f"mag-l1a-l1b-t{test_number}-cal.cdf"
    calibration_input = None
    if cdf_file.exists():
        calibration_input = load_cdf(cdf_file)

    input_mag_l1a = pd.read_csv(source_directory / f"mag-l1a-l1b-t{test_number}-in.csv")

    mag_l1a_mago = mag_l1a_dataset_generator(len(input_mag_l1a.index))
    mag_l1a_magi = mag_l1a_dataset_generator(len(input_mag_l1a.index))

    mago_vectors = np.zeros((len(input_mag_l1a.index), 4), dtype=np.int64)
    magi_vectors = np.zeros((len(input_mag_l1a.index), 4), dtype=np.int64)
    compression_flags = np.zeros((len(input_mag_l1a.index), 2))
    for index in input_mag_l1a.index:
        mago_vectors[index] = [
            input_mag_l1a["x_pri"].iloc[index],
            input_mag_l1a["y_pri"].iloc[index],
            input_mag_l1a["z_pri"].iloc[index],
            input_mag_l1a["rng_pri"].iloc[index],
        ]
        magi_vectors[index] = [
            input_mag_l1a["x_sec"].iloc[index],
            input_mag_l1a["y_sec"].iloc[index],
            input_mag_l1a["z_sec"].iloc[index],
            input_mag_l1a["rng_sec"].iloc[index],
        ]
        compression_flags[index] = [
            input_mag_l1a["compression"].iloc[index],
            input_mag_l1a["compression_width_bits"].iloc[index],
        ]

    mag_l1a_mago["vectors"].data = mago_vectors
    mag_l1a_magi["vectors"].data = magi_vectors

    mag_l1a_magi.attrs["Logical_source"] = "imap_mag_l1a_norm-magi"

    mago_epoch = MagL1a.calculate_vector_time(
        mago_vectors,
        2,
        TimeTuple(input_mag_l1a["pri_coarse"][0], input_mag_l1a["pri_fine"][0]),
    )[:, -1]
    magi_epoch = MagL1a.calculate_vector_time(
        magi_vectors,
        2,
        TimeTuple(input_mag_l1a["sec_coarse"][0], input_mag_l1a["sec_fine"][0]),
    )[:, -1]

    mag_l1a_mago.coords["epoch"] = xr.DataArray(mago_epoch)
    mag_l1a_magi.coords["epoch"] = xr.DataArray(magi_epoch)

    mag_l1a_mago["compression_flags"].data = compression_flags
    mag_l1a_magi["compression_flags"].data = compression_flags

    mago = mag_l1b(mag_l1a_mago, "v000", calibration_input)
    magi = mag_l1b(mag_l1a_magi, "v000", calibration_input)

    expected_mago = pd.read_csv(
        source_directory / f"mag-l1a-l1b-t{test_number}-mago-out.csv"
    )
    expected_magi = pd.read_csv(
        source_directory / f"mag-l1a-l1b-t{test_number}-magi-out.csv"
    )

    for index in expected_magi.index:
        assert np.allclose(
            expected_magi["x"].iloc[index],
            magi["vectors"].data[index][0],
            atol=1e-9,
            rtol=0,
        )
        assert np.allclose(
            expected_magi["y"].iloc[index],
            magi["vectors"].data[index][1],
            atol=1e-9,
            rtol=0,
        )
        assert np.allclose(
            expected_magi["z"].iloc[index],
            magi["vectors"].data[index][2],
            atol=1e-9,
            rtol=0,
        )
        assert expected_magi["range"].iloc[index] == magi["vectors"].data[index][3]
        assert (
            expected_magi["compression"].iloc[index]
            == magi["compression_flags"].data[index][0]
        )
        if expected_magi["compression"].iloc[index] != 0:
            assert (
                expected_magi["compression_width"].iloc[index]
                == magi["compression_flags"].data[index][1]
            )

        expected_time = str_to_et(expected_magi["t"].iloc[index])
        magi_time = ttj2000ns_to_et(magi["epoch"].data[index])

        assert np.allclose(expected_time, magi_time, atol=1e-6, rtol=0)

    for index in expected_mago.index:
        # TODO: come back to timestamp.
        # Can't compare UTC, coarse/fine don't work.
        assert np.allclose(
            expected_mago["x"].iloc[index],
            mago["vectors"].data[index][0],
            atol=1e-9,
            rtol=0,
        )
        assert np.allclose(
            expected_mago["y"].iloc[index],
            mago["vectors"].data[index][1],
            atol=1e-9,
            rtol=0,
        )
        assert np.allclose(
            expected_mago["z"].iloc[index],
            mago["vectors"].data[index][2],
            atol=1e-9,
            rtol=0,
        )
        assert expected_mago["range"].iloc[index] == mago["vectors"].data[index][3]
        assert (
            expected_mago["compression"].iloc[index]
            == mago["compression_flags"].data[index][0]
        )
        if expected_mago["compression"].iloc[index] != 0:
            assert (
                expected_mago["compression_width"].iloc[index]
                == mago["compression_flags"].data[index][1]
            )

        expected_time = str_to_et(expected_mago["t"].iloc[index])
        mago_time = ttj2000ns_to_et(mago["epoch"].data[index])
        assert np.allclose(expected_time, mago_time, atol=1e-6, rtol=0)
