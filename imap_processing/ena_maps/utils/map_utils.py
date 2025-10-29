"""Utilities for generating ENA maps."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def vectorized_bincount(
    indices: NDArray, weights: NDArray | None = None, minlength: int = 0
) -> NDArray:
    """
    Vectorized version of np.bincount for multi-dimensional arrays.

    This function applies np.bincount across multi-dimensional input arrays by
    adding offsets to the indices and flattening, then reshaping the result.
    This approach allows broadcasting between indices and weights.

    Parameters
    ----------
    indices : NDArray
        Array of non-negative integers to be binned. Can be multi-dimensional.
        If multi-dimensional, bincount is applied independently along each
        leading dimension.
    weights : NDArray, optional
        Array of weights that is broadcastable with indices. If provided, each
        weight is accumulated into its corresponding bin. If None (default),
        each index contributes a count of 1.
    minlength : int, optional
        Minimum number of bins in the output array. Applied to each independent
        bincount operation. Default is 0.

    Returns
    -------
    NDArray
        Array of binned values with the same leading dimensions as the input
        arrays, and a final dimension of size minlength (or the maximum index + 1,
        whichever is larger).

    See Also
    --------
    numpy.bincount : The underlying function being vectorized.

    Examples
    --------
    >>> indices = np.array([[0, 1, 1], [2, 2, 3]])
    >>> vectorized_bincount(indices, minlength=4)
    array([[1., 2., 0., 0.],
           [0., 0., 2., 1.]])
    """
    # Handle 1D case directly
    if indices.ndim == 1 and (weights is None or weights.ndim == 1):
        return np.bincount(indices, weights=weights, minlength=minlength)

    # For multi-dimensional arrays, broadcast indices and weights
    if weights is not None:
        indices_bc, weights_bc = np.broadcast_arrays(indices, weights)
        weights_flat = weights_bc.ravel()
    else:
        indices_bc = indices
        weights_flat = None

    # Get the shape for reshaping output
    non_spatial_shape = indices_bc.shape[:-1]
    n_binsets = np.prod(non_spatial_shape)

    # Determine actual minlength if not specified
    if minlength == 0:
        minlength = int(np.max(indices_bc)) + 1

    # We want to flatten the multi-dimensional bincount problem into a 1D problem.
    # This can be done by offsetting the indices for each element of each additional
    # dimension by an integer multiple of the number of bins. Doing so gives
    # each element in the additional dimensions its own set of 1D bins: index 0
    # uses bins [0, minlength), index 1 uses bins [minlength, 2*minlength), etc.
    offsets = np.arange(n_binsets).reshape(*non_spatial_shape, 1) * minlength
    indices_flat = (indices_bc + offsets).ravel()

    # Single bincount call with flattened data
    binned_flat = np.bincount(
        indices_flat, weights=weights_flat, minlength=n_binsets * minlength
    )

    # Reshape to separate each sample's bins
    binned_values = binned_flat.reshape(n_binsets, -1)[:, :minlength].reshape(
        *non_spatial_shape, minlength
    )

    return binned_values


def bin_single_array_at_indices(
    value_array: NDArray,
    projection_grid_shape: tuple[int, ...],
    projection_indices: NDArray,
    input_indices: NDArray | None = None,
    input_valid_mask: NDArray | None = None,
) -> NDArray:
    """
    Bin an array of values at the given indices.

    NOTE: The output array's spatial axis is always the final (-1) axis.

    Parameters
    ----------
    value_array : NDArray
        Array of values to bin. The final axis is the one and only spatial axis.
        If other axes are present, they will be binned independently
        along the spatial axis.
    projection_grid_shape : tuple[int, ...]
        The shape of the grid onto which values are projected
        (rows, columns) if the grid is rectangular,
        or just (number of bins,) if the grid is 1D.
    projection_indices : NDArray
        Ordered indices for projection grid, corresponding to indices in input grid.
        Can be 1-dimensional or multi-dimensional. If multi-dimensional, must be
        broadcastable with value_array. May contain non-unique indices, depending
        on the projection method.
    input_indices : NDArray
        Ordered indices for input grid, corresponding to indices in projection grid.
        1 dimensional. May be non-unique, depending on the projection method.
        If None (default), an numpy.arange of the same length as the final axis of
        value_array is used.
    input_valid_mask : NDArray, optional
        Boolean mask array for valid values in input grid.
        If None, all pixels are considered valid. Default is None.
        Must be broadcastable with value_array and projection_indices.

    Returns
    -------
    NDArray
        Binned values on the projection grid. The output shape depends on the
        input shapes after broadcasting:
        - If value_array is 1D: returns 1D array of shape (num_projection_indices,)
        - If value_array is multi-dimensional: returns array with shape
          (*value_array.shape[:-1], num_projection_indices), where the leading
          dimensions match value_array's non-spatial dimensions and the final
          dimension contains the binned values for each projection grid position.
        - If projection_indices is multi-dimensional and broadcasts with value_array,
          the output shape will be (broadcasted_shape[:-1], num_projection_indices).

    Raises
    ------
    ValueError
        If input_indices is not a 1D array, or if the arrays cannot be
        broadcast together.
    """
    # Set and check input_indices
    if input_indices is None:
        input_indices = np.arange(value_array.shape[-1])
    # input_indices must be 1D
    if input_indices.ndim != 1:
        raise ValueError(
            "input_indices must be a 1D array. "
            "If using a rectangular grid, the indices must be unwrapped."
        )

    # Verify projection_indices is broadcastable with value_array
    try:
        broadcasted_shape = np.broadcast_shapes(
            projection_indices.shape, value_array.shape
        )
    except ValueError as e:
        raise ValueError(
            f"projection_indices shape {projection_indices.shape} must be "
            f"broadcastable with value_array shape {value_array.shape}"
        ) from e

    # Set and check input_valid_mask
    if input_valid_mask is None:
        input_valid_mask = np.ones(value_array.shape[-1], dtype=bool)
    else:
        input_valid_mask = np.asarray(input_valid_mask, dtype=bool)
    # Verify input_valid_mask is broadcastable with value_array
    try:
        np.broadcast_shapes(input_valid_mask.shape, value_array.shape)
    except ValueError as e:
        raise ValueError(
            f"input_valid_mask shape {input_valid_mask.shape} must be "
            f"broadcastable with value_array shape {value_array.shape}"
        ) from e

    # Broadcast input_valid_mask to match value_array shape if needed
    input_valid_mask_bc = np.broadcast_to(input_valid_mask, broadcasted_shape)

    # Select values at input_indices positions along the spatial axis
    values = value_array[..., input_indices]

    # Apply mask: set invalid values to 0
    values_masked = np.where(input_valid_mask_bc, values, 0)

    num_projection_indices = int(np.prod(projection_grid_shape))

    # Use vectorized_bincount to handle arbitrary dimensions
    binned_values = vectorized_bincount(
        projection_indices, weights=values_masked, minlength=num_projection_indices
    )

    return binned_values


def bin_values_at_indices(
    input_values_to_bin: dict[str, NDArray],
    projection_grid_shape: tuple[int, ...],
    projection_indices: NDArray,
    input_indices: NDArray | None = None,
    input_valid_mask: NDArray | None = None,
) -> dict[str, NDArray]:
    """
    Project values from input grid to projection grid based on matched indices.

    Parameters
    ----------
    input_values_to_bin : dict[str, NDArray]
        Dict matching variable names to arrays of values to bin.
        The final (-1) axis of each array must be the one and only spatial axis,
        which the indices correspond to and on which the values will be binned.
        The other axes will be binned independently along this final spatial axis.
    projection_grid_shape : tuple[int, ...]
        The shape of the grid onto which values are projected (rows, columns).
        This size of the resulting grid (rows * columns) will be the size of the
        projected values contained in the output dictionary.
    projection_indices : NDArray
        Ordered indices for projection grid, corresponding to indices in input grid.
        1 dimensional. May be non-unique, depending on the projection method.
    input_indices : NDArray
        Ordered indices for input grid, corresponding to indices in projection grid.
        1 dimensional. May be non-unique, depending on the projection method.
        If None (default), behavior is determined by bin_single_array_at_indices.
    input_valid_mask : NDArray, optional
        Boolean mask array for valid values in input grid.
        If None, all pixels are considered valid. Default is None.

    Returns
    -------
    dict[str, NDArray]
        Dict matching the input variable names to the binned values
        on the projection grid.

    ValueError
        If the input and projection indices are not 1D arrays
        with the same number of elements.
    """
    binned_values_dict = {}
    for value_name, value_array in input_values_to_bin.items():
        logger.info(f"Binning {value_name}")
        binned_values_dict[value_name] = bin_single_array_at_indices(
            value_array=value_array,
            projection_grid_shape=projection_grid_shape,
            projection_indices=projection_indices,
            input_indices=input_indices,
            input_valid_mask=input_valid_mask,
        )

    return binned_values_dict
