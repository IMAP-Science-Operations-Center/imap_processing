import numpy as np
import pytest

from imap_processing.ena_maps.utils import map_utils


class TestENAMapMappingUtils:
    def test_bin_single_array_at_indices(
        self,
    ):
        """Test coverage for bin_single_array_at_indices function w/ simple 1D input"""
        value_array = np.array([1, 2, 3, 4, 5, 6])
        input_indices = np.array([0, 1, 2, 2, 1, 0])
        projection_indices = np.array([1, 2, 3, 1, 2, 3])
        projection_grid_shape = (5,)
        expected_projection_values = np.array([0, 4, 4, 4, 0])
        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )
        np.testing.assert_equal(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_extra_axis(
        self,
    ):
        """Test coverage for bin_single_array_at_indices function w/ simple 2D input,
        Corresponding to an extra axis that is not spatially binned.
        """
        # Binning will occur along axis 1 seen below
        # (combining 1, 2, 3 and 4, 5, 6 separately).
        value_array = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        input_indices = np.array([0, 1, 2, 2])
        projection_indices = np.array([1, 0, 1, 6])
        projection_grid_shape = (7, 1)
        expected_projection_values = np.array(
            [[2, 4, 0, 0, 0, 0, 3], [5, 10, 0, 0, 0, 0, 6]]
        )
        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )

        np.testing.assert_equal(projection_values, expected_projection_values)

    @pytest.mark.parametrize(
        "projection_grid_shape", [(1, 1), (10, 10), (180, 360), (360, 720), (360, 180)]
    )
    @pytest.mark.parametrize(
        "input_grid_shape", [(1, 1), (10, 10), (180, 360), (360, 720), (360, 180)]
    )
    def test_bin_single_array_at_indices_complex_2d(
        self, projection_grid_shape, input_grid_shape
    ):
        """Test coverage for bin_single_array_at_indices function w/ complex 2D input,
        Corresponding to an extra axis that is not spatially binned.
        Parameterized across different input and projection grid shapes.
        """
        np.random.seed(0)
        extra_axis_size = 11  # Another axis which is not spatially binned, e.g. energy
        input_grid_size = np.prod(input_grid_shape)
        projection_grid_size = np.prod(projection_grid_shape)
        value_array = np.random.rand(extra_axis_size, input_grid_size)
        input_indices = np.random.randint(0, input_grid_size, size=1000)
        projection_indices = np.random.randint(0, projection_grid_size, size=1000)
        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )

        # Explicitly check that the shape of the output is the same as projection grid
        np.testing.assert_equal(
            projection_values.shape,
            (
                extra_axis_size,
                projection_grid_size,
            ),
        )

        # Create the expected projection values by summing the input values in a loop
        # This is different from the binning function, which uses np.bincount
        expected_projection_values = np.zeros((extra_axis_size, projection_grid_size))
        for ii, ip in zip(input_indices, projection_indices):
            expected_projection_values[:, ip] += value_array[:, ii]

        np.testing.assert_allclose(projection_values, expected_projection_values)

    @pytest.mark.parametrize("projection_grid_shape", [(1, 1), (10, 10), (180, 360)])
    @pytest.mark.parametrize("input_grid_shape", [(1, 1), (10, 10), (180, 360)])
    @pytest.mark.parametrize("num_extra_dims", [1, 2, 3, 5])
    def test_bin_single_array_at_indices_complex_3d(
        self, projection_grid_shape, input_grid_shape, num_extra_dims
    ):
        """Test coverage for bin_single_array_at_indices function w/ complex N-Dim input
        Corresponding to 2 extra axes that are not spatially binned.
        Parameterized across different input and projection grid shapes.
        """
        np.random.seed(0)
        extra_axes_sizes = np.full(num_extra_dims, 3, dtype=int).tolist()
        input_grid_size = np.prod(input_grid_shape)
        projection_grid_size = np.prod(projection_grid_shape)
        value_array = np.random.rand(*extra_axes_sizes, input_grid_size)
        input_indices = np.random.randint(0, input_grid_size, size=1000)
        projection_indices = np.random.randint(0, projection_grid_size, size=1000)
        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )

        # Explicitly check that the shape of the output is the same as projection grid
        np.testing.assert_equal(
            projection_values.shape,
            (*extra_axes_sizes, projection_grid_size),
        )

        # Create the expected projection values by summing the input values in a loop
        # This is different from the binning function, which uses np.bincount
        expected_projection_values = np.zeros((*extra_axes_sizes, projection_grid_size))
        for ii, ip in zip(input_indices, projection_indices):
            expected_projection_values[..., ip] += value_array[..., ii]

        np.testing.assert_allclose(projection_values, expected_projection_values)

    # Parameterize by the size of the projection grid,
    # which is not necessarily same size as input grid
    @pytest.mark.parametrize("projection_grid_shape", [(1, 1), (10, 10), (360, 720)])
    def test_bin_values_at_indices_collapse_to_idx_zero(self, projection_grid_shape):
        """Test coverage for bin_values_at_indices function w/ dict of multiple
        1D input value arrays and a single 2D input value array.
        All input values are binned to the first index of the projection grid.
        Parameterized across different projection grid shapes.
        """
        # 1D input values (2nd will be scalar multiple of 1st)
        input_values_1d_1 = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        scale_factor_1d = 1.5
        input_values_1d_2 = input_values_1d_1 * scale_factor_1d

        # 2D input values. The 0 axis (different rows) will be summed independently
        input_values_2d = np.array(
            [
                [
                    -0.5,
                    1,
                    4,
                    7,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    2,
                    5,
                    8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0.5,
                    3,
                    6,
                    9,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        )

        extra_axis_size_2d = input_values_2d.shape[0]

        # 3D input values
        input_values_3d = np.zeros((3, 3, input_values_2d.shape[-1]))
        input_values_3d[:, :, :2] = np.array(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
                [
                    [10, 11, 12],
                    [13, 14, 15],
                    [16, 17, 18],
                ],
            ]
        ).transpose(1, 2, 0)
        extra_axes_size_3d = input_values_3d.shape[:-1]

        # Set up the expected projection values
        expected_projection_values_1d_1 = np.zeros(projection_grid_shape).ravel()
        expected_projection_values_1d_1[0] = np.sum(input_values_1d_1)
        expected_projection_values_1d_2 = (
            expected_projection_values_1d_1 * scale_factor_1d
        )
        expected_projection_values_2d = np.zeros(
            (extra_axis_size_2d, np.prod(projection_grid_shape))
        )
        expected_projection_values_2d[:, 0] = np.array([11.5, 15, 18.5])
        expected_projection_values_3d = np.zeros(
            (*extra_axes_size_3d, np.prod(projection_grid_shape))
        )
        expected_projection_values_3d[:, :, 0] = np.array(
            [
                [11, 13, 15],
                [17, 19, 21],
                [23, 25, 27],
            ]
        )

        input_values_to_bin = {
            "sum_variable_1d_1": input_values_1d_1,
            "sum_variable_1d_2": input_values_1d_2,
            "sum_variable_2d": np.array(input_values_2d),
            "sum_variable_3d": np.array(input_values_3d),
        }

        # Set up indices
        input_indices = np.arange(len(input_values_1d_1))
        projection_indices = np.zeros_like(input_indices)

        output_dict = map_utils.bin_values_at_indices(
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
            input_values_to_bin=input_values_to_bin,
            input_indices=input_indices,
        )

        np.testing.assert_equal(
            output_dict["sum_variable_1d_1"], expected_projection_values_1d_1
        )
        np.testing.assert_equal(
            output_dict["sum_variable_1d_2"], expected_projection_values_1d_2
        )
        np.testing.assert_equal(
            output_dict["sum_variable_2d"], expected_projection_values_2d
        )
        np.testing.assert_equal(
            output_dict["sum_variable_3d"], expected_projection_values_3d
        )

    def test_bin_values_at_indices_2d_indices_raises(self):
        """2D indices are not supported for binning.
        Test that ValueError is raised."""
        input_values = np.array([1, 2, 3])
        input_indices = np.array([[0, 1], [1, 2]])
        projection_indices = np.array([0, 1, 2])
        projection_grid_shape = (3,)

        with pytest.raises(
            ValueError,
            match=(
                "Indices must be 1D arrays. If using a rectangular grid, "
                "the indices must be unwrapped."
            ),
        ):
            map_utils.bin_single_array_at_indices(
                input_values,
                input_indices=input_indices,
                projection_indices=projection_indices,
                projection_grid_shape=projection_grid_shape,
            )

    def test_bin_values_at_indices_mismatched_sizes_raises(self):
        """Mismatched input and projection indices should raise an error.
        Test that ValueError is raised."""
        input_values = np.array([1, 2, 3])
        input_indices = np.array([0, 1, 0, 1])
        projection_indices = np.array([0, 1, 2])
        projection_grid_shape = (3,)

        with pytest.raises(
            ValueError,
            match=("The number of input and projection indices must be the same"),
        ):
            map_utils.bin_single_array_at_indices(
                input_values,
                input_indices=input_indices,
                projection_indices=projection_indices,
                projection_grid_shape=projection_grid_shape,
            )
