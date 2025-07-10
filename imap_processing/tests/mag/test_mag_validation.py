from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.ancillary.ancillary_dataset_combiner import MagAncillaryCombiner
from imap_processing.cdf.utils import load_cdf
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1a.mag_l1a import mag_l1a
from imap_processing.mag.l1a.mag_l1a_data import MagL1a, TimeTuple
from imap_processing.mag.l1b.mag_l1b import mag_l1b
from imap_processing.mag.l1c.mag_l1c import mag_l1c
from imap_processing.mag.l2.mag_l2 import mag_l2
from imap_processing.spice.time import (
    TTJ2000_EPOCH,
    et_to_ttj2000ns,
    str_to_et,
    ttj2000ns_to_et,
)
from imap_processing.tests.mag.conftest import (
    mag_generate_l1b_from_csv,
    mag_l1a_dataset_generator,
)


@pytest.mark.parametrize(
    ("test_number"), ["001", "002", "003", "004", "005", "006", "007", "008"]
)
def test_mag_l1a_validation(test_number):
    source_directory = Path(__file__).parent / "validation" / "L1a" / f"T{test_number}"

    input_file = source_directory / f"mag-l0-l1a-t{test_number}-in.bin"
    expected_output_file = source_directory / f"mag-l0-l1a-t{test_number}-out.csv"

    mag_l1a_out = mag_l1a(input_file)
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
def test_mag_l1b_validation(test_number, mocks):
    source_directory = Path(__file__).parent / "validation" / "L1b" / f"T{test_number}"

    # used in most tests
    default_cal = (
        Path(__file__).parent
        / "validation"
        / "calibration"
        / "imap_mag_l1b-calibration_20240229_v002.cdf"
    )
    test_cal = (
        source_directory / f"imap_mag_l1a-l1b-t{test_number}-cal_20240201_v001.cdf"
    )

    if test_cal.exists():
        mocks["construct_path"].return_value = test_cal
        combined = MagAncillaryCombiner([test_cal], np.datetime64("2025-03-01"))
    else:
        mocks["construct_path"].return_value = default_cal
        combined = MagAncillaryCombiner([default_cal], np.datetime64("2025-03-01"))

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

    expected_mago = pd.read_csv(
        source_directory / f"mag-l1a-l1b-t{test_number}-mago-out.csv"
    )
    expected_magi = pd.read_csv(
        source_directory / f"mag-l1a-l1b-t{test_number}-magi-out.csv"
    )
    day = np.datetime64(expected_magi["t"].iloc[0]).astype("datetime64[D]")
    mago = mag_l1b(mag_l1a_mago, day, combined.combined_dataset)
    magi = mag_l1b(mag_l1a_magi, day, combined.combined_dataset)

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


@pytest.mark.xfail(reason="All L1C edge cases are not yet complete")
@pytest.mark.parametrize(("test_number"), ["013", "014", "015", "016"])
@pytest.mark.parametrize(("sensor"), ["mago", "magi"])
@pytest.mark.usefixtures("_download_test_data")
def test_mag_l1c_validation(test_number, sensor):
    # We expect tests 013 and 014 to pass. 015 and 016 are not yet complete.
    # timestamp = (
    #     (np.datetime64("2025-03-11T12:22:50.706034") - np.datetime64(TTJ2000_EPOCH))
    #     / np.timedelta64(1, "ns")
    # ).astype(np.int64)
    # print(f"Time stamp shift: {timestamp }")

    source_directory = Path(__file__).parent / "validation" / "L1c" / f"T{test_number}"
    norm_in = source_directory / f"mag-l1b-l1c-t{test_number}-{sensor}-normal-in.csv"
    burst_in = source_directory / f"mag-l1b-l1c-t{test_number}-{sensor}-burst-in.csv"

    norm = mag_generate_l1b_from_csv(
        pd.read_csv(norm_in), f"imap_mag_l1b_norm-{sensor}"
    )
    burst = mag_generate_l1b_from_csv(
        pd.read_csv(burst_in), f"imap_mag_l1b_burst-{sensor}"
    )

    # out = np.int64(794968123760272000)
    # print(f"expected out {TTJ2000_EPOCH + out.astype('timedelta64[ns]')}")
    # For mago test 013: norm 2, burst 64
    norm.attrs["vectors_per_second"] = get_vecsec(test_number, sensor, "norm")

    burst.attrs["vectors_per_second"] = get_vecsec(test_number, sensor, "burst")

    l1c = mag_l1c(norm, burst)
    expected_output = pd.read_csv(
        source_directory / f"mag-l1b-l1c-t{test_number}-{sensor}-normal-out.csv"
    )

    for index in expected_output.index:
        assert np.allclose(
            expected_output["x"].iloc[index],
            l1c["vectors"].data[index][0],
            atol=1e-4,
            rtol=0,
        )
        assert np.allclose(
            expected_output["y"].iloc[index],
            l1c["vectors"].data[index][1],
            atol=1e-4,
            rtol=0,
        )
        assert np.allclose(
            expected_output["z"].iloc[index],
            l1c["vectors"].data[index][2],
            atol=1e-4,
            rtol=0,
        )

        expected_time = np.datetime64(expected_output["t"].iloc[index])
        l1c_time = TTJ2000_EPOCH + l1c["epoch"].data[index].astype("timedelta64[ns]")
        assert expected_time - l1c_time < np.timedelta64(500, "ms")


@pytest.mark.parametrize(("test_number", "mode"), [("021", "burst"), ("022", "norm")])
@pytest.mark.usefixtures("_download_test_data")
def test_mag_l2_validation(test_number, mode):
    source_directory = Path(__file__).parent / "validation" / "L2" / f"T{test_number}"
    magi_in = source_directory / f"mag-l1bc-l2-t{test_number}-mago-{mode}-in.csv"

    # Generate L1B files
    magi = mag_generate_l1b_from_csv(pd.read_csv(magi_in), f"imap_mag_l1b_{mode}-magi")

    file = source_directory / "imap_mag_l2-calibration-matrices_20250506_v005.cdf"

    # Mock AncillaryFilePath.construct_path() to point to source_directory
    with patch("imap_data_access.AncillaryFilePath.construct_path", return_value=file):
        # Use MagAncillaryCombiner to create calibration dataset from input file
        calibration_dataset = MagAncillaryCombiner(
            [file],
            np.datetime64("2025-05-07"),
        ).combined_dataset

    # Get offsets dataset out of the validation directory
    offsets_dataset = load_cdf(
        source_directory / f"imap_mag_l2_{mode}-offsets_20250506_v006.cdf"
    )

    offsets_dataset = offsets_dataset.assign_coords(epoch=magi["epoch"])

    # Run the L2 processing
    l2 = mag_l2(
        calibration_dataset,
        offsets_dataset,
        magi,
        np.datetime64("2025-05-06"),
        DataMode[mode.upper()],
    )[0]

    # Compare to expected output
    output_file = source_directory / f"imap_mag_l2_{mode}_20250506_v007.csv"

    expected_output = pd.read_csv(output_file)
    expected_output["epoch"] = pd.to_datetime(expected_output["epoch"])
    # Expected output timestamps are in UTC which means the timestamps include
    # leapseconds.
    # Subtract 5 seconds from epoch to account for leap seconds
    expected_output["epoch"] = expected_output["epoch"] - pd.Timedelta(seconds=5)

    # Truncate expected output to only include time after 2025-05-06T00:00:00
    expected_output = expected_output[
        expected_output["epoch"] >= np.datetime64("2025-05-06T00:00:00")
    ]

    for index in range(len(expected_output)):
        expected_time = et_to_ttj2000ns(
            str_to_et(str(expected_output["epoch"].iloc[index]))
        )
        l2_time = l2["epoch"].data[index]

        assert abs(expected_time - l2_time) < 200000

        assert np.allclose(
            expected_output["x"].iloc[index],
            l2["vectors"].data[index][0],
            atol=1e-5,
            rtol=0,
        )
        assert np.allclose(
            expected_output["y"].iloc[index],
            l2["vectors"].data[index][1],
            atol=1e-5,
            rtol=0,
        )
        assert np.allclose(
            expected_output["z"].iloc[index],
            l2["vectors"].data[index][2],
            atol=1e-5,
            rtol=0,
        )


def get_vecsec(test_number, sensor, mode):
    # Manually pulled from MAG validation test PDF, which describes the input
    # sensor rates for each test.

    # values are equal to start_time:vector rate for test data files.
    # in production this will be passed up from L1B.
    # TODO: fill in from PDF file
    vecsec = {
        "013": {
            "mago": {"norm": "794966559703707008:2", "burst": "794966835183206016:64"},
            "magi": {"norm": "794966559703691008:2", "burst": "794966835198801024:8"},
        },
        "014": {
            "mago": {"norm": "795154763219339008:2", "burst": "795155024191415040:128"},
            "magi": {"norm": "795154763219369984:2", "burst": "795155024191446016:128"},
        },
        "015": {
            "mago": {
                "norm": "794967514703783040:2,794968123760272000:4",
                "burst": "794966835183206016:64",
            },
            "magi": {
                "norm": "794967514703768064:2,794968123760256000:1",
                "burst": "794967834198914944:64",
            },
        },
        "016": {"mago": {"norm": "", "burst": ""}, "magi": {"norm": "", "burst": ""}},
    }
    return vecsec[test_number][sensor][mode]
