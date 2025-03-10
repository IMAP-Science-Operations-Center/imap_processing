from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.mag.l1b.mag_l1b import (
    calibrate_vector,
    mag_l1b,
    mag_l1b_processing,
    rescale_vector,
)
from imap_processing.tests.mag.conftest import (
    mag_l1a_dataset_generator,
)


def test_mag_processing(mag_test_calibration_data):
    # All specific test values come from MAG team to accommodate various cases.
    # Each vector is multiplied by the matrix in the calibration data for the given
    # range to get the calibrated vector.
    mag_l1a_dataset = mag_l1a_dataset_generator(20)
    mag_l1a_dataset["compression_flags"].data[1, :] = np.array([1, 18], dtype=np.int8)

    mag_l1a_dataset["vectors"].data[0, :] = np.array([1, 1, 1, 0])
    mag_l1a_dataset["vectors"].data[1, :] = np.array([7982, 48671, -68090, 0])
    mag_attributes = ImapCdfAttributes()
    mag_attributes.add_instrument_global_attrs("mag")
    mag_attributes.add_instrument_variable_attrs("mag", "l1b")
    mag_l1b = mag_l1b_processing(
        mag_l1a_dataset,
        mag_test_calibration_data,
        mag_attributes,
        "imap_mag_l1b_norm-mago",
    )

    np.testing.assert_allclose(
        mag_l1b["vectors"][0].values, [2.2972, 2.2415, 2.2381, 0], atol=1e-4
    )
    np.testing.assert_allclose(
        mag_l1b["vectors"][1].values,
        [4584.1029091, 27238.73161294, -38405.22240195, 0.0],
    )

    np.testing.assert_allclose(mag_l1b["vectors"][2].values, [0, 0, 0, 0])

    assert mag_l1b["vectors"].values.shape == mag_l1a_dataset["vectors"].values.shape

    mag_l1b = mag_l1b_processing(
        mag_l1a_dataset,
        mag_test_calibration_data,
        mag_attributes,
        "imap_mag_l1b_norm-magi",
    )

    np.testing.assert_allclose(
        mag_l1b["vectors"][0].values, [2.27538, 2.23416, 2.23682, 0], atol=1e-5
    )

    assert mag_l1b["vectors"].values.shape == mag_l1a_dataset["vectors"].values.shape


def test_mag_attributes():
    mag_l1a_dataset = mag_l1a_dataset_generator(20)

    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1a_norm-mago"]

    output = mag_l1b(mag_l1a_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1b_norm-mago"

    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1a_burst-magi"]

    output = mag_l1b(mag_l1a_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1b_burst-magi"

    assert output.attrs["Data_level"] == "L1B"


def test_cdf_output():
    l1a_cdf = load_cdf(
        Path(__file__).parent
        / "validation"
        / "imap_mag_l1a_norm-magi_20251017_v001.cdf"
    )
    l1b_dataset = mag_l1b(l1a_cdf, "v001")

    output_path = write_cdf(l1b_dataset)

    assert Path.exists(output_path)


def test_mag_compression_scale():
    mag_l1a_dataset = mag_l1a_dataset_generator(20)

    test_calibration = np.array(
        [
            [2.2972202, 0.0, 0.0],
            [0.00348625, 2.23802879, 0.0],
            [-0.00250788, -0.00888437, 2.24950008],
        ]
    )
    mag_l1a_dataset["vectors"][0, :] = np.array([1, 1, 1, 0])
    mag_l1a_dataset["vectors"][1, :] = np.array([1, 1, 1, 0])
    mag_l1a_dataset["vectors"][2, :] = np.array([1, 1, 1, 0])
    mag_l1a_dataset["vectors"][3, :] = np.array([1, 1, 1, 0])

    mag_l1a_dataset["compression_flags"][0, :] = np.array([1, 16], dtype=np.int8)
    mag_l1a_dataset["compression_flags"][1, :] = np.array([0, 0], dtype=np.int8)
    mag_l1a_dataset["compression_flags"][2, :] = np.array([1, 18], dtype=np.int8)
    mag_l1a_dataset["compression_flags"][3, :] = np.array([1, 14], dtype=np.int8)

    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1a_norm-mago"]
    output = mag_l1b(mag_l1a_dataset, "v001")

    calibrated_vectors = np.matmul(test_calibration, np.array([1, 1, 1]))
    # 16 bit width is the standard
    assert np.allclose(output["vectors"].data[0][:3], calibrated_vectors)
    # uncompressed data is uncorrected
    assert np.allclose(output["vectors"].data[1][:3], calibrated_vectors)

    # width of 18 should be multiplied by 1/4
    scaled_vectors = calibrated_vectors * 1 / 4
    # should be corrected
    assert np.allclose(output["vectors"].data[2][:3], scaled_vectors)

    # width of 14 should be multiplied by 4
    scaled_vectors = calibrated_vectors * 4
    assert np.allclose(output["vectors"].data[3][:3], scaled_vectors)


def test_rescale_vector():
    # From algo document examples
    vector = np.array([10, -2000, 0])
    expected_vector = np.array([2.5, -500, 0])
    output = rescale_vector(vector, [1, 18])
    assert np.allclose(output, expected_vector)

    vector = np.array([32766, -2, 1])
    expected_vector = np.array([65532, -4, 2])
    output = rescale_vector(vector, [1, 15])
    assert np.allclose(output, expected_vector)


def test_calibrate_vector():
    # from MFOTOURFO
    cal_array = np.array(
        [
            [
                [2.29722020e00, 7.38200160e-02, 1.88479865e-02, 4.59777333e-03],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            ],
            [
                [3.48624576e-03, 1.09224000e-04, 3.26118600e-05, 5.02830000e-06],
                [2.23802879e00, 7.23781440e-02, 1.84842873e-02, 4.50744060e-03],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            ],
            [
                [-2.50787532e-03, -8.33760000e-05, -2.71240200e-05, 2.50509000e-06],
                [-8.88437262e-03, -2.84256000e-04, -7.41600000e-05, -2.29399200e-05],
                [2.24950008e00, 7.23836160e-02, 1.84847323e-02, 4.50945192e-03],
            ],
        ]
    )

    calibration_matrix = xr.DataArray(cal_array)
    # All cal vector comparisons were calculated by hand and confirmed by MAG team.

    cal_vector = calibrate_vector(np.array([1.0, 1.0, 1.0, 0]), calibration_matrix)

    expected_vector = np.array([2.2972, 2.2415, 2.2381, 0])

    assert np.allclose(cal_vector, expected_vector, atol=1e-9)

    cal_vector = calibrate_vector(np.array([1.1, -2.0, 3.0, 1]), calibration_matrix)
    expected_vector = np.array([(0.081202, -0.144636, 0.217628, 1)])
    assert np.allclose(cal_vector, expected_vector, atol=1e-9)

    cal_vector = calibrate_vector(
        rescale_vector(np.array([7982, 48671, -68090, 0]), (1, 18)), calibration_matrix
    )
    expected_vector = [4584.1029091, 27238.73161294, -38405.22240195, 0.0]

    assert np.allclose(cal_vector, expected_vector, atol=1e-9)


def test_l1a_to_l1b(validation_l1a):
    # Convert l1a input validation packet file to l1b
    with pytest.raises(ValueError, match="Raw L1A"):
        mag_l1b(validation_l1a[0], "v000")

    l1b = [mag_l1b(i, "v000") for i in validation_l1a[1:]]

    assert len(l1b) == len(validation_l1a) - 1

    assert l1b[0].attrs["Logical_source"] == "imap_mag_l1b_norm-mago"
    assert l1b[1].attrs["Logical_source"] == "imap_mag_l1b_norm-magi"

    assert len(l1b[0]["vectors"].data) > 0
    assert len(l1b[1]["vectors"].data) > 0
