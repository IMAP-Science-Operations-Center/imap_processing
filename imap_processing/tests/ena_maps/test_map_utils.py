import numpy as np
import pytest

from imap_processing.ena_maps.utils import map_utils


class TestVectorizedBincount:
    def test_vectorized_bincount_1d(self):
        """Test vectorized_bincount with 1D input (equivalent to np.bincount)."""
        indices = np.array([0, 1, 1, 2, 2, 2])
        result = map_utils.vectorized_bincount(indices, minlength=4)
        expected = np.array([1.0, 2.0, 3.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_vectorized_bincount_1d_with_weights(self):
        """Test vectorized_bincount with 1D input and weights."""
        indices = np.array([0, 1, 1, 2, 2, 2])
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = map_utils.vectorized_bincount(indices, weights=weights, minlength=4)
        expected = np.array([1.0, 5.0, 15.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_vectorized_bincount_2d(self):
        """Test vectorized_bincount with 2D input (multiple 1D bincounts)."""
        indices = np.array([[0, 1, 1], [2, 2, 3]])
        result = map_utils.vectorized_bincount(indices, minlength=4)
        expected = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0]])
        np.testing.assert_array_equal(result, expected)

    def test_vectorized_bincount_2d_with_weights(self):
        """Test vectorized_bincount with 2D input and weights."""
        indices = np.array([[0, 1, 1], [2, 2, 3]])
        weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = map_utils.vectorized_bincount(indices, weights=weights, minlength=4)
        expected = np.array([[1.0, 5.0, 0.0, 0.0], [0.0, 0.0, 9.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_vectorized_bincount_3d(self):
        """Test vectorized_bincount with 3D input."""
        indices = np.array([[[0, 1], [1, 2]], [[2, 3], [3, 0]]])
        result = map_utils.vectorized_bincount(indices, minlength=4)
        expected = np.array(
            [
                [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_vectorized_bincount_no_minlength(self):
        """Test vectorized_bincount without specifying minlength."""
        indices = np.array([[0, 1, 1], [2, 2, 3]])
        result = map_utils.vectorized_bincount(indices)
        # Without minlength, output size is max(indices) + 1 = 4
        expected = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0]])
        np.testing.assert_array_equal(result, expected)


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
        # value_array has shape (2, 4) - 2 energy bins, 4 spatial positions
        # Binning will occur along the spatial axis (axis 1)
        value_array = np.array(
            [
                [1, 2, 3, 4],
                [10, 20, 30, 40],
            ]
        )
        # Select input positions 0, 1, 2, 3 and map to projection positions
        input_indices = np.array([0, 1, 2, 3])
        projection_indices = np.array([1, 0, 1, 6])
        projection_grid_shape = (7,)
        # Row 0: proj[0]=2, proj[1]=1+3=4, proj[6]=4
        # Row 1: proj[0]=20, proj[1]=10+30=40, proj[6]=40
        expected_projection_values = np.array(
            [[2, 4, 0, 0, 0, 0, 4], [20, 40, 0, 0, 0, 0, 40]]
        )
        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )

        np.testing.assert_equal(projection_values, expected_projection_values)

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

    def test_bin_single_array_at_indices_2d_input_indices_raises(self):
        """2D input_indices are not supported for binning.
        Test that ValueError is raised."""
        input_values = np.array([1, 2, 3])
        input_indices = np.array([[0, 1], [1, 2]])
        projection_indices = np.array([0, 1, 2])
        projection_grid_shape = (3,)

        with pytest.raises(
            ValueError,
            match=(
                "input_indices must be a 1D array. If using a rectangular grid, "
                "the indices must be unwrapped."
            ),
        ):
            map_utils.bin_single_array_at_indices(
                input_values,
                input_indices=input_indices,
                projection_indices=projection_indices,
                projection_grid_shape=projection_grid_shape,
            )

    def test_bin_single_array_at_indices_multidim_projection_indices(self):
        """Test bin_single_array_at_indices multi-dimensional projection_indices."""
        # 2D value_array with shape (3, 4) - 3 energy bins, 4 spatial positions
        value_array = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
                [100.0, 200.0, 300.0, 400.0],
            ]
        )
        # 2D projection_indices with shape (3, 4) - different mapping per energy
        projection_indices = np.array([[0, 1, 1, 2], [1, 0, 2, 2], [2, 2, 1, 0]])
        input_indices = np.array([0, 1, 2, 3])
        projection_grid_shape = (3,)

        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )

        # Expected output shape: (3, 3) - 3 energy bins, 3 projection bins
        # Energy 0: [[0,0]=1, [0,1]+[0,2]=2+3=5, [0,3]=4]
        # Energy 1: [[1,1]=20, [1,0]=10, [1,2]+[1,3]=30+40=70]
        # Energy 2: [[2,3]=400, [2,2]=300, [2,0]+[2,1]=100+200=300]
        expected_projection_values = np.array(
            [[1.0, 5.0, 4.0], [20.0, 10.0, 70.0], [400.0, 300.0, 300.0]]
        )

        np.testing.assert_equal(projection_values.shape, (3, 3))
        np.testing.assert_allclose(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_broadcasting(self):
        """Test broadcasting between 1D projection_indices and 2D value_array."""
        # 2D value_array with shape (2, 4)
        value_array = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        # 1D projection_indices with shape (3,) - broadcasts to (2, 3)
        projection_indices = np.array([0, 1, 0])
        input_indices = np.array([0, 1, 2])
        projection_grid_shape = (2,)

        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
        )

        # Expected: both rows use same projection_indices
        # Row 0: bin 0 gets value[0,0]+value[0,2]=1+3=4, bin 1 gets value[0,1]=2
        # Row 1: bin 0 gets value[1,0]+value[1,2]=10+30=40, bin 1 gets value[1,1]=20
        expected_projection_values = np.array([[4.0, 2.0], [40.0, 20.0]])

        np.testing.assert_equal(projection_values.shape, (2, 2))
        np.testing.assert_allclose(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_with_1d_mask(self):
        """Test bin_single_array_at_indices with 1D input_valid_mask."""
        value_array = np.array([1, 2, 3, 4, 5, 6])
        input_indices = np.array([0, 1, 2, 2, 1, 0])
        projection_indices = np.array([1, 2, 3, 1, 2, 3])
        projection_grid_shape = (5,)
        # Mask out indices 1 and 4 (values 2 and 5)
        input_valid_mask = np.array([True, False, True, True, False, False])

        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
            input_valid_mask=input_valid_mask,
        )

        # Without mask: [0, 4, 4, 4, 0]
        # With mask (excluding indices 1,4,5 -> values 2,2,1): [0, 4, 0, 3, 0]
        expected_projection_values = np.array([0, 4, 0, 3, 0])
        np.testing.assert_equal(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_with_2d_mask(self):
        """
        Test bin_single_array_at_indices with 2D input_valid_mask.

        input_valid_mask matches value_array shape.
        """
        # 2D value_array with shape (2, 6)
        value_array = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [10, 20, 30, 40, 50, 60],
            ]
        )
        input_indices = np.array([0, 1, 2, 2, 1, 0])
        projection_indices = np.array([1, 2, 3, 1, 2, 3])
        projection_grid_shape = (5,)
        # Mask with different patterns for each row
        input_valid_mask = np.array(
            [
                [True, False, True, True, False, True],  # Row 0: mask out 2, 5
                [True, True, False, True, True, False],  # Row 1: mask out 3, 6
            ]
        )

        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
            input_valid_mask=input_valid_mask,
        )

        # Row 0: mask excludes values 2,5 -> [0, 4, 0, 4, 0]
        # Row 1: mask excludes values 30,60 -> [0, 40, 40, 0, 0]
        expected_projection_values = np.array([[0, 4, 0, 4, 0], [0, 40, 40, 0, 0]])
        np.testing.assert_equal(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_with_broadcast_mask(self):
        """Test bin_single_array_at_indices with 1D mask, 2D value_array."""
        # 2D value_array with shape (2, 6)
        value_array = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [10, 20, 30, 40, 50, 60],
            ]
        )
        input_indices = np.array([0, 1, 2, 2, 1, 0])
        projection_indices = np.array([1, 2, 3, 1, 2, 3])
        projection_grid_shape = (5,)
        # 1D mask that broadcasts to both rows
        input_valid_mask = np.array([True, False, True, True, False, True])

        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
            input_valid_mask=input_valid_mask,
        )

        # Same mask applied to both rows: exclude indices 1,4
        # Row 0: mask excludes values 2,5 -> [0, 4, 0, 4, 0]
        # Row 1: mask excludes values 30,60 -> [0, 40, 0, 40, 0]
        expected_projection_values = np.array([[0, 4, 0, 4, 0], [0, 40, 0, 40, 0]])
        np.testing.assert_equal(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_mask_all_invalid(self):
        """Test bin_single_array_at_indices with all values masked out."""
        value_array = np.array([1, 2, 3, 4, 5, 6])
        input_indices = np.array([0, 1, 2, 2, 1, 0])
        projection_indices = np.array([1, 2, 3, 1, 2, 3])
        projection_grid_shape = (5,)
        # Mask out all values
        input_valid_mask = np.array([False, False, False, False, False, False])

        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            input_indices=input_indices,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
            input_valid_mask=input_valid_mask,
        )

        # All values masked -> all zeros
        expected_projection_values = np.array([0, 0, 0, 0, 0])
        np.testing.assert_equal(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_mask_shape_mismatch_raises(self):
        """Test that ValueError is raised when shapes are incompatible."""
        value_array = np.array([1, 2, 3, 4, 5, 6])
        input_indices = np.array([0, 1, 2])
        projection_indices = np.array([1, 2, 3])
        projection_grid_shape = (5,)
        # Incompatible mask shape
        input_valid_mask = np.array([True, False, True])

        with pytest.raises(
            ValueError,
            match="projection_indices shape .* must be broadcastable "
            "with value_array shape",
        ):
            map_utils.bin_single_array_at_indices(
                value_array,
                input_indices=input_indices,
                projection_indices=projection_indices,
                projection_grid_shape=projection_grid_shape,
                input_valid_mask=input_valid_mask,
            )
        with pytest.raises(
            ValueError,
            match="input_valid_mask shape .* must be broadcastable "
            "with value_array shape",
        ):
            map_utils.bin_single_array_at_indices(
                value_array,
                input_indices=input_indices,
                projection_indices=np.arange(6),
                projection_grid_shape=projection_grid_shape,
                input_valid_mask=input_valid_mask,
            )

    def test_bin_single_array_at_indices_mask_default_input_indices(self):
        """Test bin_single_array_at_indices with mask and default input_indices=None."""
        # Test that when input_indices is not provided, the function uses
        # np.arange(value_array.shape[-1]) and masking works correctly
        value_array = np.array([1, 2, 3, 4, 5, 6])
        projection_indices = np.array([0, 1, 1, 2, 2, 0])
        projection_grid_shape = (3,)
        # Mask out values at positions 1, 3, 5 (values 2, 4, 6)
        input_valid_mask = np.array([True, False, True, False, True, False])

        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
            input_valid_mask=input_valid_mask,
        )

        # Without mask: bin[0]=1+6=7, bin[1]=2+3=5, bin[2]=4+5=9
        # With mask: bin[0]=1+0=1, bin[1]=0+3=3, bin[2]=0+5=5
        expected_projection_values = np.array([1, 3, 5])
        np.testing.assert_equal(projection_values, expected_projection_values)

    def test_bin_single_array_at_indices_mask_default_input_indices_2d(self):
        """Test with mask, default input_indices, and 2D value_array."""
        # 2D value_array with shape (2, 6)
        value_array = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [10, 20, 30, 40, 50, 60],
            ]
        )
        projection_indices = np.array([0, 1, 1, 2, 2, 0])
        projection_grid_shape = (3,)
        # 1D mask broadcasting to both rows
        input_valid_mask = np.array([True, False, True, False, True, False])

        projection_values = map_utils.bin_single_array_at_indices(
            value_array,
            projection_indices=projection_indices,
            projection_grid_shape=projection_grid_shape,
            input_valid_mask=input_valid_mask,
        )

        # Row 0: bin[0]=1, bin[1]=3, bin[2]=5
        # Row 1: bin[0]=10, bin[1]=30, bin[2]=50
        expected_projection_values = np.array([[1, 3, 5], [10, 30, 50]])
        np.testing.assert_equal(projection_values, expected_projection_values)
