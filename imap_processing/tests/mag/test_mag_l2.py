from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf
from imap_processing.mag.l2.mag_l2 import apply_calibration_matrix, mag_l2
from imap_processing.tests.mag.conftest import mag_l1a_dataset_generator


@pytest.fixture()
def norm_dataset():
    dataset = mag_l1a_dataset_generator(20)
    epoch_vals = np.arange(0, 10, 0.5) * 1e9
    vectors_per_second_attr = "0:2,4000000000:4"
    dataset.attrs["vectors_per_second"] = vectors_per_second_attr
    dataset["epoch"] = epoch_vals
    dataset.attrs["Logical_source"] = "imap_mag_l1c_norm-mago"
    vectors = np.array([[i, i, i, 2] for i in range(1, 21)])
    dataset["vectors"].data = vectors

    return dataset


def test_mag_l2(norm_dataset):
    calibration_dataset = load_cdf(
        Path(__file__).parent
        / "validation"
        / "calibration"
        / "imap_mag_l1b-calibration_20240229_v001.cdf"
    )
    offset_dataset = xr.Dataset()
    l2 = mag_l2(calibration_dataset, offset_dataset, norm_dataset, "v001")
    assert "vectors" in l2[0].data_vars


def test_failure_on_mismatch_files():
    # input_offsets =
    pass


def test_apply_calibration(norm_dataset, mag_test_l1b_calibration_data):
    # Test matrix application
    output = apply_calibration_matrix(
        np.array([[1, 1, 1, 0]], dtype=np.float64), mag_test_l1b_calibration_data, True
    )

    expected_vector = np.array([2.2972, 2.2415, 2.2381, 0])

    assert np.allclose(output, expected_vector, atol=1e-9)

    vectors = np.array([[1, 1, 1, 0] for i in range(1, 21)], dtype=np.float64)
    expected_output = np.array([[2.2972, 2.2415, 2.2381, 0] for i in range(1, 21)])
    output = apply_calibration_matrix(vectors, mag_test_l1b_calibration_data, True)

    assert np.allclose(output, expected_output, atol=1e-9)


def test_offset_application():
    # test offsets
    pass


def test_full_calculation():
    # test matrix + offsets calculation
    pass


def test_timestamp_truncation():
    # Test that data is truncated to exactly 24 hours
    pass


def test_fail_on_missing_offsets():
    # Processing should fail if vectors do not have corresponding timestamps
    pass


def test_magnitude():
    # Test magnitude calculation
    pass


def test_expected_output_norm():
    # should return 4 files with correct attributes
    pass


def test_expected_output_burst():
    # should return 4 files with correct attributes
    pass
