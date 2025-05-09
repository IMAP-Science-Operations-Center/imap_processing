import numpy as np
import pytest
import xarray as xr

from imap_processing.mag import imap_mag_sdc_configuration_v001 as configuration
from imap_processing.mag.constants import VecSec
from imap_processing.mag.l1c.interpolation_methods import (
    InterpolationFunction,
    cic_filter,
    estimate_rate,
)
from imap_processing.mag.l1c.mag_l1c import (
    fill_normal_data,
    find_all_gaps,
    find_gaps,
    generate_timeline,
    interpolate_gaps,
    mag_l1c,
    process_mag_l1c,
    vectors_per_second_from_string,
)
from imap_processing.tests.mag.conftest import (
    generate_test_epoch,
    mag_l1a_dataset_generator,
)


@pytest.fixture(scope="module")
def mag_l1b_dataset():
    output_dataset = mag_l1a_dataset_generator(10)

    output_dataset["epoch"] = xr.DataArray(
        np.arange(0.1, 5.1, step=0.5) * 1e9, name="epoch", dims=["epoch"]
    )
    vectors = np.array([[i, i, i, 2] for i in range(1, 11)])
    vectors[0, :] = np.array([1, 1, 1, 0])
    output_dataset["vectors"].data = vectors

    return output_dataset


@pytest.fixture
def norm_dataset():
    dataset = mag_l1a_dataset_generator(10)
    epoch_vals = generate_test_epoch(
        6,
        [
            VecSec.TWO_VECS_PER_S,
            VecSec.FOUR_VECS_PER_S,
            VecSec.FOUR_VECS_PER_S,
        ],
        0,
        [[2, 4], [4.25, 5.5]],
    )
    vectors_per_second_attr = "0:2,4000000000:4"
    dataset.attrs["vectors_per_second"] = vectors_per_second_attr
    dataset["epoch"] = epoch_vals
    dataset.attrs["Logical_source"] = "imap_mag_l1b_norm-mago"
    vectors = np.array([[i, i, i, 2] for i in range(1, 11)])
    dataset["vectors"].data = vectors

    return dataset


@pytest.fixture
def burst_dataset():
    dataset = mag_l1a_dataset_generator(27)
    epoch_vals = generate_test_epoch(5.1, [VecSec.EIGHT_VECS_PER_S], 1.9)
    dataset["epoch"] = epoch_vals
    dataset.attrs["Logical_source"] = ["imap_mag_l1b_burst-mago"]
    vectors = np.array([[i, i, i, 2] for i in range(1, 28)])
    dataset["vectors"].data = vectors

    vectors_per_second_attr = "0:8"
    dataset.attrs["vectors_per_second"] = vectors_per_second_attr

    return dataset


def test_configuration_file():
    assert configuration.L1C_INTERPOLATION_METHOD in [
        e.name for e in InterpolationFunction
    ]


def test_interpolation_methods():
    # very basic test of all methods
    vectors = np.random.rand(200, 4)
    input_timestamps = np.arange(0, 50, step=0.25) * 1e9
    output_timestamps = np.arange(10, 20, step=0.5) * 1e9
    for method in InterpolationFunction:
        output = method(
            vectors,
            input_timestamps,
            output_timestamps,
            input_rate=VecSec.FOUR_VECS_PER_S,
            output_rate=VecSec.TWO_VECS_PER_S,
        )
        assert len(output) == 20
        output = method(
            vectors,
            input_timestamps,
            output_timestamps,
            input_rate=None,
            output_rate=None,
        )
        assert len(output) == 20


def test_process_mag_l1c(norm_dataset, burst_dataset):
    l1c = process_mag_l1c(norm_dataset, burst_dataset, InterpolationFunction.linear)
    expected_output_timeline = (
        np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.25, 4.75, 5.25, 5.5, 5.75, 6])
        * 1e9
    )
    assert np.array_equal(l1c[:, 0], expected_output_timeline)
    # Last new timestamp is missing data because burst mode only goes to 5.15
    # Don't generate data if there's no burst data to interpolate
    assert (
        np.count_nonzero([np.sum(l1c[i, 1:4]) for i in range(l1c.shape[0])])
        == l1c.shape[0] - 1
    )
    expected_flags = np.zeros(15)
    # filled sections should have 1 as a flag
    expected_flags[5:8] = 1
    expected_flags[10:11] = 1
    # last datapoint in the gap is missing a value
    expected_flags[11] = -1
    assert np.array_equal(l1c[:, 5], expected_flags)
    assert np.array_equal(l1c[:5, 1:5], norm_dataset["vectors"].data[:5, :])
    for i in range(5, 8):
        e = l1c[i, 0]
        burst_vectors = burst_dataset.sel(epoch=int(e), method="nearest")[
            "vectors"
        ].data
        # We're just finding the closest burst values to the array, so they won't be
        # identical.
        assert np.allclose(l1c[i, 1:5], burst_vectors, rtol=0, atol=1)

    assert np.array_equal(l1c[8:10, 1:5], norm_dataset["vectors"].data[5:7, :])
    for i in range(10, 11):
        e = l1c[i, 0]
        burst_vectors = burst_dataset.sel(epoch=int(e), method="nearest")[
            "vectors"
        ].data
        # We're just finding the closest burst values to the array, so they won't be
        # identical.
        assert np.allclose(l1c[i, 1:5], burst_vectors, rtol=0, atol=1)

    assert np.array_equal(l1c[11, 1:5], [0, 0, 0, 0])


def test_interpolate_gaps(norm_dataset, mag_l1b_dataset):
    # np.array([0, 0.5, 1, 1.5, 2, 4, 4.25, 5.5, 5.75, 6]) * 1e9
    gaps = np.array([[2 * 1e9, 4 * 1e9, 2], [4.25 * 1e9, 5.5 * 1e9, 2]])
    generated_timeline = generate_timeline(norm_dataset["epoch"].data, gaps)
    norm_timeline = fill_normal_data(norm_dataset, generated_timeline)
    gaps = np.array([[2 * 1e9, 4 * 1e9, 2]])
    output = interpolate_gaps(
        mag_l1b_dataset, gaps, norm_timeline, InterpolationFunction.linear
    )
    expected_output = np.array(
        [
            [5.8, 5.8, 5.8, 2, 1, 0, 0],
            [6.8, 6.8, 6.8, 2, 1, 0, 0],
            [7.8, 7.8, 7.8, 2, 1, 0, 0],
        ]
    )

    assert np.allclose(output[5:8, 1:], expected_output)

    input_norm_timeline = np.array(
        [
            [1.50e09, 4, 4, 4, 2, 0, 0, 0],
            [2.00e09, 5, 5, 5, 2, 0, 0, 0],
            [2.50e09, 0, 0, 0, 0, 1, 0, 0],
            [3.00e09, 0, 0, 0, 0, 1, 0, 0],
            [3.50e09, 0, 0, 0, 0, 1, 0, 0],
            [4.00e09, 6, 6, 6, 2, 0, 0, 0],
            [4.25e09, 7, 7, 7, 2, 0, 0, 0],
        ]
    )

    # output - all timestamps with -1 should be filled with interpolated values.
    expected_output = np.array(
        [
            [1.50e09, 4, 4, 4, 2, 0, 0, 0],
            [2.00e09, 5, 5, 5, 2, 0, 0, 0],
            [2.50e09, 5.8, 5.8, 5.8, 2, 1, 0, 0],
            [3.00e09, 6.8, 6.8, 6.8, 2, 1, 0, 0],
            [3.50e09, 7.8, 7.8, 7.8, 2, 1, 0, 0],
            [4.00e09, 6, 6, 6, 2, 0, 0, 0],
            [4.25e09, 7, 7, 7, 2, 0, 0, 0],
        ]
    )

    output = interpolate_gaps(
        mag_l1b_dataset, gaps, input_norm_timeline, InterpolationFunction.linear
    )

    assert np.allclose(output, expected_output)


def test_mag_l1c(norm_dataset, burst_dataset):
    l1c = mag_l1c(burst_dataset, norm_dataset)
    assert l1c["vector_magnitude"].shape == (len(l1c["epoch"].data),)
    assert l1c["vector_magnitude"].data[0] == np.linalg.norm(l1c["vectors"].data[0][:4])
    assert l1c["vector_magnitude"].data[-1] == np.linalg.norm(
        l1c["vectors"].data[-1][:4]
    )

    expected_vars = [
        "vectors",
        "compression_flags",
        "vector_magnitude",
        "generated_flag",
    ]

    for var in expected_vars:
        assert var in l1c.data_vars


def test_mag_attributes(norm_dataset, burst_dataset):
    output = mag_l1c(norm_dataset, burst_dataset)
    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-mago"

    expected_attrs = ["missing_sequences", "interpolation_method"]
    for attr in expected_attrs:
        assert attr in output.attrs


def test_missing_burst_file(norm_dataset, burst_dataset):
    # Should run with only normal mode data or only burst mode data.
    output = mag_l1c(norm_dataset, None)
    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-mago"

    # Should pass through normal mode data only
    assert np.array_equal(output["vectors"].data, norm_dataset["vectors"].data)
    assert np.array_equal(output["epoch"].data, norm_dataset["epoch"].data)


@pytest.mark.xfail(reason="Burst mode only not implemented yet")
def test_missing_norm_file(norm_dataset, burst_dataset):
    # Should run with only normal mode data or only burst mode data.
    burst_dataset.attrs["Logical_source"] = "imap_mag_l1b_burst-magi"
    output = mag_l1c(burst_dataset, None)

    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-magi"
    # TODO: test that the output is downsampled
    # TODO: How to test against previous day's file?


def test_find_all_gaps():
    epoch_test = generate_test_epoch(
        5.5, [VecSec.TWO_VECS_PER_S, VecSec.TWO_VECS_PER_S], 0, [[2, 5]]
    )

    vectors_per_second = vectors_per_second_from_string("0:2")

    output = find_all_gaps(epoch_test, vectors_per_second)
    expected_gaps = np.array([[2 * 1e9, 5 * 1e9, 2]])
    assert np.array_equal(output, expected_gaps)

    epoch_test = np.array([0, 0.5, 1, 1.5, 2, 4, 4.25, 4.5, 4.75, 5.5]) * 1e9
    vectors_per_second_attr = vectors_per_second_from_string("0:2,4000000000:4")
    expected_gaps = np.array([[2 * 1e9, 4 * 1e9, 2], [4.75 * 1e9, 5.5 * 1e9, 4]])
    output = find_all_gaps(epoch_test, vectors_per_second_attr)
    assert np.array_equal(output, expected_gaps)


def test_find_gaps():
    # Test should be in ns
    epoch_test = generate_test_epoch(
        3.5, [VecSec.TWO_VECS_PER_S], 0, [[0.5, 2], [2, 3.5]]
    )
    gaps = find_gaps(epoch_test, 2)
    expected_return = np.array([[0.5 * 1e9, 2 * 1e9, 2], [2 * 1e9, 3.5 * 1e9, 2]])

    assert np.array_equal(gaps, expected_return)

    epoch_test = generate_test_epoch(
        5, [VecSec.TWO_VECS_PER_S], gaps=[[0.5, 2], [3, 4]]
    )
    gaps = find_gaps(epoch_test, 2)
    expected_return = np.array([[0.5 * 1e9, 2 * 1e9, 2], [3 * 1e9, 4 * 1e9, 2]])

    assert np.array_equal(gaps, expected_return)

    epoch_test = generate_test_epoch(
        3, [VecSec.FOUR_VECS_PER_S], gaps=[[0.5, 1], [2, 3]]
    )
    gaps = find_gaps(epoch_test, 4)
    expected_return = np.array([[0.5 * 1e9, 1 * 1e9, 4], [2 * 1e9, 3 * 1e9, 4]])

    assert np.array_equal(gaps, expected_return)


def test_generate_timeline():
    epoch_test = generate_test_epoch(
        3, [VecSec.FOUR_VECS_PER_S], gaps=[[0.5, 1], [2, 3]]
    )

    gaps = np.array([[0.5, 1], [2, 3]]) * 1e9
    expected_output = np.array([0, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]) * 1e9
    output = generate_timeline(epoch_test, gaps)
    assert np.array_equal(output, expected_output)

    epoch_test = generate_test_epoch(5, [VecSec.TWO_VECS_PER_S], starting_point=1)
    # Expected output from find_gaps if none are found
    gaps = np.zeros((0, 2))
    output = generate_timeline(epoch_test, gaps)
    assert np.array_equal(output, epoch_test)

    epoch_test = generate_test_epoch(
        5, [VecSec.TWO_VECS_PER_S], starting_point=1, gaps=[[3, 5]]
    )
    gaps = np.array([[3, 5]]) * 1e9

    expected_output = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]) * 1e9
    output = generate_timeline(epoch_test, gaps)
    assert np.array_equal(output, expected_output)


def test_fill_normal_data(mag_l1b_dataset):
    output_timeline = np.arange(0.1, 6.1, step=0.5) * 1e9
    output = fill_normal_data(mag_l1b_dataset, output_timeline)

    assert output.shape == (12, 8)
    # all vectors should be nonzero
    assert np.count_nonzero(output[:-2, 1:4]) == 30
    # last two vectors should be zero
    assert np.count_nonzero(output[-2:, 1:5]) == 0

    # spot check
    assert np.array_equal(output[0, 1:5], mag_l1b_dataset["vectors"].data[0, :])
    assert np.array_equal(output[5, 1:5], mag_l1b_dataset["vectors"].data[5, :])
    assert np.array_equal(output[9, 1:5], mag_l1b_dataset["vectors"].data[9, :])


def test_cic_filter():
    input_vectors = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
        ]
    )
    input_timestamps = generate_test_epoch(2, [VecSec.FOUR_VECS_PER_S], 0)
    output_timestamps = generate_test_epoch(2, [VecSec.TWO_VECS_PER_S], 0)

    output = cic_filter(
        input_vectors,
        input_timestamps,
        output_timestamps,
        VecSec.FOUR_VECS_PER_S,
        VecSec.TWO_VECS_PER_S,
    )
    assert len(output) != 0
    # TODO: How to test this?


def test_estimate_rate():
    input_timestamps = generate_test_epoch(2, [VecSec.FOUR_VECS_PER_S], 0)
    output_timestamps = generate_test_epoch(2, [VecSec.TWO_VECS_PER_S], 0)

    input = estimate_rate(input_timestamps)
    assert input == VecSec.FOUR_VECS_PER_S

    output = estimate_rate(output_timestamps)
    assert output == VecSec.TWO_VECS_PER_S
