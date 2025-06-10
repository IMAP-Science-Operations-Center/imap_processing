import numpy as np
import pytest
import xarray as xr

from imap_processing.mag.constants import DataMode
from imap_processing.mag.l2.mag_l2 import mag_l2, retrieve_matrix_from_l2_calibration
from imap_processing.mag.l2.mag_l2_data import MagL2
from imap_processing.spice.time import et_to_datetime64, et_to_utc, ttj2000ns_to_et
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
    l2 = mag_l2(
        calibration_dataset, offset_dataset, norm_dataset, np.datetime64("2025-10-17")
    )
    assert "vectors" in l2[0].data_vars


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


@pytest.mark.xfail(reason="Error is too strict during testing")
def test_error_raises(mag_test_l2_data):
    dataset = mag_l1a_dataset_generator(3504)
    with pytest.raises(ValueError, match="same timestamps"):
        mag_l2(
            mag_test_l2_data[0],
            mag_test_l2_data[1],
            dataset,
            np.datetime64("2025-10-17"),
        )

    dataset = mag_l1a_dataset_generator(3505)
    with pytest.raises(ValueError, match="same timestamps"):
        mag_l2(
            mag_test_l2_data[0],
            mag_test_l2_data[1],
            dataset,
            np.datetime64("2025-10-17"),
        )


@pytest.mark.parametrize(
    ("time_shift", "start_diff", "end_diff"),
    # 3 hours in ns
    [
        (-1.08e13, -1, 0),
        # 19 hours in ns
        (6.84e13, 0, 1),
    ],
)
def test_timestamp_truncation(
    norm_dataset, mag_test_l2_data, time_shift, start_diff, end_diff
):
    day = np.datetime64("2025-10-17").astype("datetime64[D]")
    shifted_timestamps = norm_dataset["epoch"].data + time_shift
    l2 = MagL2(
        norm_dataset["vectors"].data[:, :3],
        shifted_timestamps,
        norm_dataset["vectors"].data[:, 3],
        {},
        np.zeros(len(norm_dataset["epoch"].data)),
        np.zeros(len(norm_dataset["epoch"].data)),
        DataMode.NORM,
        offsets=np.zeros((len(norm_dataset["epoch"].data), 3)),
        timedelta=np.zeros(len(norm_dataset["epoch"].data)),
    )
    first_epoch_val = np.array(et_to_utc(ttj2000ns_to_et(l2.epoch[0]))).astype(
        "datetime64[D]"
    )

    # Before starting: epoch spans two days
    assert first_epoch_val == day + start_diff

    last_epoch_val = np.array(et_to_utc(ttj2000ns_to_et(l2.epoch[-1]))).astype(
        "datetime64[D]"
    )
    assert last_epoch_val == day + end_diff

    l2.truncate_to_24h(day)

    # after truncation: epoch spans one day
    first_epoch_val = et_to_datetime64(ttj2000ns_to_et(l2.epoch[0])).astype(
        "datetime64[D]"
    )
    last_epoch_val = et_to_datetime64(ttj2000ns_to_et(l2.epoch[-1])).astype(
        "datetime64[D]"
    )

    assert first_epoch_val == day
    assert last_epoch_val == day

    # Timestamps should align with all data
    assert l2.epoch.shape[0] == l2.vectors.shape[0]
    assert l2.epoch.shape[0] == l2.magnitude.shape[0]
    assert l2.epoch.shape[0] == l2.range.shape[0]
    assert l2.epoch.shape[0] == l2.quality_flags.shape[0]
    assert l2.epoch.shape[0] == l2.quality_bitmask.shape[0]

    assert l2.epoch.shape[0] < shifted_timestamps.shape[0]
    post_trunc_shape = l2.epoch.shape[0]

    for ts in l2.epoch:
        assert ts in shifted_timestamps
    # Applying twice shouldn't affect anything
    l2.truncate_to_24h(day)

    assert l2.epoch.shape[0] == post_trunc_shape

    first_epoch_val = et_to_datetime64(ttj2000ns_to_et(l2.epoch[0])).astype(
        "datetime64[D]"
    )
    last_epoch_val = et_to_datetime64(ttj2000ns_to_et(l2.epoch[-1])).astype(
        "datetime64[D]"
    )

    assert first_epoch_val == day
    assert last_epoch_val == day


def test_magnitude():
    # Test magnitude calculation
    test_vector_one = np.array([[6, 9, 12]])
    expected_magnitude = np.sqrt(6**2 + 9**2 + 12**2)

    output_magnitude = MagL2.calculate_magnitude(test_vector_one)

    assert np.allclose(output_magnitude, expected_magnitude, atol=1e-9)

    test_multiple_vectors = np.random.rand(10, 3) * 10
    expected_magnitude = [
        np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) for x in test_multiple_vectors
    ]

    output_magnitude = MagL2.calculate_magnitude(test_multiple_vectors)
    assert np.allclose(output_magnitude, expected_magnitude, atol=1e-9)

    assert output_magnitude.shape == (10,)


def test_expected_output_norm(norm_dataset):
    # should return 4 files with correct attributes
    # TODO: complete with L2 attributes

    pass


def test_expected_output_burst():
    # should return 4 files with correct attributes
    pass


@pytest.mark.parametrize(
    ("is_mago", "data_var"),
    [
        (True, "URFTOORFO"),
        (False, "URFTOORFI"),
    ],
)
def test_retrieve_matrix_from_l2_calibration(is_mago, data_var):
    start_day = np.datetime64("2025-10-15").astype("datetime64[D]")
    end_day = np.datetime64("2025-10-20").astype("datetime64[D]")
    epoch_vars = xr.DataArray(
        np.arange(start_day, end_day, dtype="datetime64[D]"),
        dims=["epoch"],
        coords={"epoch": np.arange(5)},
    )
    example_calibration_dataset = xr.Dataset(
        {
            "URFTOORFO": xr.DataArray(
                np.random.rand(5, 3, 3, 4),
                dims=["epoch", "URFTOORFO_dim_0", "URFTOORFO_dim_1", "URFTOORFO_dim_2"],
            ),
            "URFTOORFI": xr.DataArray(
                np.random.rand(5, 3, 3, 4),
                dims=["epoch", "URFTOORFI_dim_0", "URFTOORFI_dim_1", "URFTOORFI_dim_2"],
            ),
        },
        coords={"epoch": epoch_vars},
    )

    calibration_matrix = retrieve_matrix_from_l2_calibration(
        example_calibration_dataset, start_day, use_mago=is_mago
    )

    assert calibration_matrix.shape == (3, 3, 4)
    assert np.array_equal(
        example_calibration_dataset.sel(epoch=start_day)[data_var].data,
        calibration_matrix,
    )

    test_day = np.datetime64("2025-10-17").astype("datetime64[D]")
    calibration_matrix = retrieve_matrix_from_l2_calibration(
        example_calibration_dataset, test_day, use_mago=is_mago
    )

    assert calibration_matrix.shape == (3, 3, 4)
    assert np.array_equal(
        example_calibration_dataset.sel(epoch=test_day)[data_var].data,
        calibration_matrix,
    )
