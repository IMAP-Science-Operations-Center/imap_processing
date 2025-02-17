"""Utilities for generating ENA maps."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def bin_single_array_at_indices(
    value_array: NDArray,
    projection_grid_shape: tuple[int, int],
    projection_indices: NDArray,
    input_indices: NDArray | None = None,
) -> NDArray:
    """
    Bin an array of values at the given indices.

    Parameters
    ----------
    value_array : NDArray
        Array of values to bin. The 0th axis must be the one and only spatial axis.
        If other axes are present, they will be binned independently
        along the 0th (spatial) axis.
    projection_grid_shape : tuple[int]
        The shape of the grid onto which values are projected
        (rows, columns) if the grid is rectangular,
        or just (number of bins,) if the grid is 1D.
    projection_indices : NDArray
        Ordered indices for projection grid, corresponding to indices in input grid.
        1 dimensional. May be non-unique, depending on the projection method.
    input_indices : NDArray
        Ordered indices for input grid, corresponding to indices in projection grid.
        1 dimensional. May be non-unique, depending on the projection method.
        If None (default), an arange of the same length as the
        0th axis of value_array is used.

    Returns
    -------
    NDArray
        Binned values on the projection grid.

    Raises
    ------
    ValueError
        If the input and projection indices are not 1D arrays
        with the same number of elements.
    NotImplementedError
        If the input value_array has dimensionality less than 1.
    """
    if input_indices is None:
        input_indices = np.arange(value_array.shape[0])

    # Both sets of indices must be 1D with the same number of elements
    if input_indices.ndim != 1 or projection_indices.ndim != 1:
        raise ValueError(
            "Indices must be 1D arrays. "
            "If using a rectangular grid, the indices must be unwrapped."
        )
    if input_indices.size != projection_indices.size:
        raise ValueError(
            "The number of input and projection indices must be the same. \n"
            f"Received {input_indices.size} input indices and {projection_indices.size}"
            " projection indices."
        )

    num_projection_indices = np.prod(projection_grid_shape)

    if value_array.ndim == 1:
        binned_values = np.bincount(
            projection_indices,
            weights=value_array[input_indices],
            minlength=num_projection_indices,
        )
    elif value_array.ndim >= 2:
        # Apply bincount to each row independently
        binned_values = np.apply_along_axis(
            lambda x: np.bincount(
                projection_indices,
                weights=x[input_indices, ...],
                minlength=num_projection_indices,
            ),
            axis=0,
            arr=value_array,
        )
    else:
        raise NotImplementedError(
            "Only 1+ Dimensional arrays are supported for binning. "
            f"Received array with shape {value_array.shape}."
        )
    return binned_values


def bin_values_at_indices(
    input_values_to_bin: dict[str, NDArray],
    projection_grid_shape: tuple[int, int],
    projection_indices: NDArray,
    input_indices: NDArray | None = None,
) -> dict[str, NDArray]:
    """
    Project values from input grid to projection grid based on matched indices.

    Parameters
    ----------
    input_values_to_bin : dict[str, NDArray]
        Dict matching variable names to arrays of values to bin.
        The 0th axis of each array must be the one and only spatial axis,
        which the indices correspond to and on which the values will be binned.
        The other axes will be binned independently along this 0th axis.
    projection_grid_shape : tuple[int, int]
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
        )

    return binned_values_dict
