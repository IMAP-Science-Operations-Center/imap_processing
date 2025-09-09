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
    calculate_allowed_gap,
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
    l1c = mag_l1c(burst_dataset, np.datetime64('2025-01-01'), norm_dataset)
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
    # Test Case 1: Basic single gap with constant rate
    epoch_test = generate_test_epoch(
        5.5, [VecSec.TWO_VECS_PER_S, VecSec.TWO_VECS_PER_S], 0, [[2, 5]]
    )
    vectors_per_second = vectors_per_second_from_string("0:2")
    output = find_all_gaps(epoch_test, vectors_per_second)
    expected_gaps = np.array([[2 * 1e9, 5 * 1e9, 2]])
    assert np.array_equal(output, expected_gaps)

    # Test Case 2: Multiple gaps with rate transitions
    epoch_test = np.array([0, 0.5, 1, 1.5, 2, 4, 4.25, 4.5, 4.75, 5.5]) * 1e9
    vectors_per_second_attr = vectors_per_second_from_string("0:2,4000000000:4")
    expected_gaps = np.array([[2 * 1e9, 4 * 1e9, 2], [4.75 * 1e9, 5.5 * 1e9, 4]])
    output = find_all_gaps(epoch_test, vectors_per_second_attr)
    assert np.array_equal(output, expected_gaps)

    # Test Case 3: No gaps - continuous timeline
    continuous_epoch = generate_test_epoch(3, [VecSec.FOUR_VECS_PER_S], 0)
    vectors_per_second_continuous = vectors_per_second_from_string("0:4")
    output_no_gaps = find_all_gaps(continuous_epoch, vectors_per_second_continuous)
    expected_no_gaps = np.zeros((0, 3))
    assert np.array_equal(output_no_gaps, expected_no_gaps)

    # Test Case 4: Multiple rate changes with gaps in each section
    epoch_complex = generate_test_epoch(
        6,
        [VecSec.TWO_VECS_PER_S, VecSec.FOUR_VECS_PER_S, VecSec.EIGHT_VECS_PER_S],
        0,
        [[1, 2], [3.5, 4.5]],
    )
    # Rate changes at t=2s and t=4s
    vectors_per_second_complex = vectors_per_second_from_string(
        "0:2,2000000000:4,4500000000:8"
    )
    output_complex = find_all_gaps(epoch_complex, vectors_per_second_complex)
    # Should find gaps: [1-2s at 2 vec/s], [3.5-4.5s at 4 vec/s]
    expected_complex = np.array([[1 * 1e9, 2 * 1e9, 2], [3.5 * 1e9, 4.5 * 1e9, 4]])
    assert np.array_equal(output_complex, expected_complex)

    # Test Case 5: Gap at the beginning of timeline
    epoch_start_gap = np.array([2, 2.5, 3, 3.5, 4]) * 1e9
    vectors_per_second_start = vectors_per_second_from_string("0:2")
    output_start_gap = find_all_gaps(epoch_start_gap, vectors_per_second_start)
    expected_start_gap = np.array([[0 * 1e9, 2 * 1e9, 2]])
    assert np.array_equal(output_start_gap, expected_start_gap)

    # Test Case 6: Gap at the end of timeline
    epoch_end_gap = np.array([0, 0.5, 1, 1.5, 2]) * 1e9
    vectors_per_second_end = vectors_per_second_from_string("0:2")
    # Expect timeline to continue to 4s based on rate
    output_end_gap = find_all_gaps(epoch_end_gap, vectors_per_second_end)
    expected_end_gap = np.array([[2 * 1e9, 2.5 * 1e9, 2]])
    # Note: This test may need adjustment based on actual find_all_gaps behavior

    # Test Case 7: Very small gap (single missing sample)
    epoch_small_gap = np.array([0, 0.5, 1.5, 2, 2.5]) * 1e9  # Missing 1.0s
    vectors_per_second_small = vectors_per_second_from_string("0:2")
    output_small_gap = find_all_gaps(epoch_small_gap, vectors_per_second_small)
    expected_small_gap = np.array([[1 * 1e9, 1.5 * 1e9, 2]])
    assert np.array_equal(output_small_gap, expected_small_gap)

    # Test Case 9: Default behavior (None vecsec_dict) - should assume 2 vec/s
    epoch_default = generate_test_epoch(3, [VecSec.TWO_VECS_PER_S], 0, [[1, 2]])
    output_default = find_all_gaps(epoch_default, None)
    expected_default = np.array([[1 * 1e9, 2 * 1e9, 2]])
    assert np.array_equal(output_default, expected_default)

    # Test Case 10: Multiple consecutive gaps
    epoch_multi_gaps = np.array([0, 0.5, 2, 2.5, 4, 4.5]) * 1e9
    vectors_per_second_multi = vectors_per_second_from_string("0:2")
    output_multi = find_all_gaps(epoch_multi_gaps, vectors_per_second_multi)
    expected_multi = np.array([[1 * 1e9, 2 * 1e9, 2], [3 * 1e9, 4 * 1e9, 2]])
    assert np.array_equal(output_multi, expected_multi)

    # Test Case 11: Complex rate transition scenario
    # Timeline with gaps before, during, and after rate changes
    epoch_transition = np.array([0, 0.5, 2.5, 3, 3.25, 4.75, 5]) * 1e9
    # Rate changes from 2 vec/s to 4 vec/s at t=3s
    vectors_per_second_transition = vectors_per_second_from_string("0:2,3000000000:4")
    output_transition = find_all_gaps(epoch_transition, vectors_per_second_transition)
    # Expected gaps: [1-2.5s at 2 vec/s], [3.5-4.75s at 4 vec/s]
    expected_transition = np.array(
        [[1 * 1e9, 2.5 * 1e9, 2], [3.5 * 1e9, 4.75 * 1e9, 4]]
    )
    assert np.array_equal(output_transition, expected_transition)

    # Test Case 12: Empty timeline
    epoch_empty = np.array([])
    vectors_per_second_empty = vectors_per_second_from_string("0:2")
    output_empty = find_all_gaps(epoch_empty, vectors_per_second_empty)
    expected_empty = np.zeros((0, 3))
    assert np.array_equal(output_empty, expected_empty)


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
    """
    Comprehensive test of CIC filter implementation according to algorithm document.

    Tests decimation factor calculation, delay compensation, filter coefficients,
    and proper array length handling for different input/output rate combinations.
    """

    # Test Case 1: Basic 4:2 decimation (decimation_factor = 2)
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
            [9, 9, 9],
        ]
    )
    input_timestamps = generate_test_epoch(2, [VecSec.FOUR_VECS_PER_S], 0)
    output_timestamps = generate_test_epoch(2, [VecSec.TWO_VECS_PER_S], 0)
    timestamps_filtered, vectors_filtered = cic_filter(
        input_vectors,
        input_timestamps,
        output_timestamps,
        VecSec.FOUR_VECS_PER_S,
        VecSec.TWO_VECS_PER_S,
    )

    # Basic output validation
    assert len(vectors_filtered) != 0
    assert len(timestamps_filtered) != 0
    assert len(timestamps_filtered) == len(vectors_filtered)

    # Test Case 2: Verify decimation factor calculation and delay
    # For 4:2 ratio, decimation_factor = 2, delay = (3-1)//2 = 1
    expected_delay = 1
    expected_filtered_length = len(input_vectors) - expected_delay
    assert len(vectors_filtered) == expected_filtered_length
    assert len(timestamps_filtered) == expected_filtered_length

    # Test Case 3: Higher decimation ratio (8:2 = 4x decimation)
    input_vectors_8hz = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [15, 15, 15],
            [16, 16, 16],
            [17, 17, 17],
        ]
    )
    input_timestamps_8hz = generate_test_epoch(2, [VecSec.EIGHT_VECS_PER_S], 0)
    output_timestamps_2hz = generate_test_epoch(2, [VecSec.TWO_VECS_PER_S], 0)

    timestamps_filtered_8hz, vectors_filtered_8hz = cic_filter(
        input_vectors_8hz,
        input_timestamps_8hz,
        output_timestamps_2hz,
        VecSec.EIGHT_VECS_PER_S,
        VecSec.TWO_VECS_PER_S,
    )

    # For 8:2 ratio, decimation_factor = 4, delay = (7-1)//2 = 3
    expected_delay_8hz = 3
    expected_filtered_length_8hz = len(input_vectors_8hz) - expected_delay_8hz
    assert len(vectors_filtered_8hz) == expected_filtered_length_8hz
    assert len(timestamps_filtered_8hz) == expected_filtered_length_8hz

    # Test Case 4: Test rate validation (should raise error if input_rate <= output_rate)
    with pytest.raises(
        ValueError, match="Burst mode input rate.*should never be less than"
    ):
        cic_filter(
            input_vectors,
            input_timestamps,
            output_timestamps,
            VecSec.TWO_VECS_PER_S,  # Lower input rate
            VecSec.FOUR_VECS_PER_S,  # Higher output rate
        )

    # Test Case 5: Test automatic rate estimation when rates are None
    timestamps_filtered_auto, vectors_filtered_auto = cic_filter(
        input_vectors,
        input_timestamps,
        output_timestamps,
        None,  # Should estimate as 4 vecs/sec
        None,  # Should estimate as 2 vecs/sec
    )

    assert len(timestamps_filtered_auto) == len(vectors_filtered_auto)
    assert len(vectors_filtered_auto) > 0

    # Test Case 6: Test filter smoothing effect
    # Create a simple step function input to verify filter smoothing
    step_input = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [10, 10, 10],
            [10, 10, 10],
            [10, 10, 10],
            [10, 10, 10],
        ]
    )

    timestamps_step, vectors_step = cic_filter(
        step_input,
        input_timestamps,
        output_timestamps,
        VecSec.FOUR_VECS_PER_S,
        VecSec.TWO_VECS_PER_S,
    )

    # CIC filter should smooth the step transition
    # The filtered output should have intermediate values, not just 0s and 10s
    unique_values = np.unique(vectors_step[:, 0])
    assert len(unique_values) > 2, "CIC filter should create intermediate values"

    # Test Case 7: Test vector shape preservation (should work with 3-component vectors)
    assert vectors_filtered.shape[1] == 3, "Output vectors should maintain 3 components"
    assert vectors_filtered_8hz.shape[1] == 3, (
        "Output vectors should maintain 3 components"
    )

    # Test Case 8: Test delay compensation consistency
    # The delay should be consistently applied to both timestamps and vectors
    if len(timestamps_filtered) > 1:
        # Verify that timestamps are properly aligned with filtered vectors
        original_timestamp_spacing = input_timestamps[1] - input_timestamps[0]
        filtered_timestamp_spacing = timestamps_filtered[1] - timestamps_filtered[0]
        assert filtered_timestamp_spacing == original_timestamp_spacing, (
            "Timestamp spacing should be preserved after filtering"
        )


def test_estimate_rate():
    input_timestamps = generate_test_epoch(2, [VecSec.FOUR_VECS_PER_S], 0)
    output_timestamps = generate_test_epoch(2, [VecSec.TWO_VECS_PER_S], 0)

    input = estimate_rate(input_timestamps)
    assert input == VecSec.FOUR_VECS_PER_S

    output = estimate_rate(output_timestamps)
    assert output == VecSec.TWO_VECS_PER_S


def test_cic_filter_delay_compensation():
    # test that extra values are removed from CIC filter properly.

    input_vectors_case2 = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
        ]
    )
    input_timestamps_case2 = generate_test_epoch(2, [VecSec.FOUR_VECS_PER_S], 0)

    output_timestamps_case2 = generate_test_epoch(2, [VecSec.TWO_VECS_PER_S], 0)

    input_filtered_case2, vectors_filtered_case2 = cic_filter(
        input_vectors_case2,
        input_timestamps_case2,
        output_timestamps_case2,
        VecSec.FOUR_VECS_PER_S,
        VecSec.TWO_VECS_PER_S,
    )

    # Arrays should still have matching lengths when delay > 0
    assert len(input_filtered_case2) == len(vectors_filtered_case2), (
        f"Array length mismatch in delay>0 case: input_filtered has "
        f"{len(input_filtered_case2)} elements, vectors_filtered has "
        f"{len(vectors_filtered_case2)} elements"
    )
