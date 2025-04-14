import numpy as np
import pytest

from imap_processing.mag.l2.mag_l2 import mag_l2
from imap_processing.mag.l2.mag_l2_data import MagL2
from imap_processing.tests.mag.conftest import mag_l1a_dataset_generator


@pytest.fixture
def norm_dataset(mag_test_l2_data):
    offsets = mag_test_l2_data[1]
    dataset = mag_l1a_dataset_generator(3504)
    epoch_vals = offsets["epoch"].data
    vectors_per_second_attr = "0:2,4000000000:4"
    dataset.attrs["vectors_per_second"] = vectors_per_second_attr
    dataset["epoch"] = epoch_vals
    dataset.attrs["Logical_source"] = "imap_mag_l1c_norm-mago"
    vectors = np.array([[i, i, i, 2] for i in range(1, 3505)])
    dataset["vectors"].data = vectors

    return dataset


def test_mag_l2(norm_dataset, mag_test_l2_data):
    calibration_dataset = mag_test_l2_data[0]

    offset_dataset = mag_test_l2_data[1]
    l2 = mag_l2(calibration_dataset, offset_dataset, norm_dataset)
    assert "vectors" in l2[0].data_vars


def test_failure_on_mismatch_files():
    # input_offsets =
    pass


def test_offset_application(norm_dataset, mag_test_l2_data):
    # Test against zeros
    offsets = mag_test_l2_data[1]
    output = MagL2(
        norm_dataset["vectors"].data[:, :3],
        norm_dataset["epoch"].data,
        norm_dataset["vectors"].data[:, 3],
        {},
        None,
        None,
        None,
        offsets=offsets["offsets"].data,
        timedelta=offsets["timedeltas"].data,
    )

    expected_vectors = norm_dataset["vectors"].data[:, :3]
    assert np.allclose(output.vectors, expected_vectors, atol=1e-9)
    assert np.allclose(output.epoch, norm_dataset["epoch"], atol=1e-9)

    new_offsets = np.zeros((len(norm_dataset["epoch"]), 3))
    new_offsets[0] = [1, 1, 1]
    new_offsets[1] = [-1, -1, -1]
    new_offsets[-1] = [1, 0, -1]

    new_timeshift = np.zeros(len(norm_dataset["epoch"]))
    new_timeshift[0] = 0.00001
    new_timeshift[1] = -0.00001
    new_timeshift[2] = 1e-9

    expected_timeshift = norm_dataset["epoch"].data
    # Timeshift is provided in seconds, epoch is in nanoseconds
    expected_timeshift[0] = expected_timeshift[0] + 10000
    expected_timeshift[1] = expected_timeshift[1] - 10000
    expected_timeshift[2] = expected_timeshift[2] + 1

    output = MagL2(
        norm_dataset["vectors"].data[:, :3],
        norm_dataset["epoch"].data,
        norm_dataset["vectors"].data[:, 3],
        {},
        None,
        None,
        None,
        offsets=new_offsets,
        timedelta=new_timeshift,
    )

    expected_vectors = norm_dataset["vectors"].data[:, :3]
    expected_vectors[0] = [2, 2, 2]
    expected_vectors[1] = [1, 1, 1]
    expected_vectors[-1] = [3505, 3504, 3503]

    assert np.allclose(output.vectors, expected_vectors, atol=1e-9)
    assert np.allclose(output.epoch, expected_timeshift, atol=1e-9)


def test_error_raises(mag_test_l2_data):
    dataset = mag_l1a_dataset_generator(3504)
    with pytest.raises(ValueError, match="same timestamps"):
        mag_l2(mag_test_l2_data[0], mag_test_l2_data[1], dataset)

    dataset = mag_l1a_dataset_generator(3505)
    with pytest.raises(ValueError, match="same timestamps"):
        mag_l2(mag_test_l2_data[0], mag_test_l2_data[1], dataset)


def test_full_calculation(norm_dataset, mag_test_l2_data):
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
