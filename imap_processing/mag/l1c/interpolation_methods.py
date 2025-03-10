from enum import Enum

import numpy as np
from scipy.interpolate import make_interp_spline


def linear(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
):
    print("linear")
    spline = make_interp_spline(input_timestamps, input_vectors, k=1)
    return spline(output_timestamps)


def quadratic(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
):
    spline = make_interp_spline(input_timestamps, input_vectors, k=2)
    return spline(output_timestamps)


def cubic(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
):
    spline = make_interp_spline(input_timestamps, input_vectors, k=3)
    return spline(output_timestamps)


def linear_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
):
    pass


def quadratic_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
):
    pass


def cubic_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
):
    pass


class InterpolationFunction(Enum):
    linear = (linear,)
    quadratic = (quadratic,)
    cubic = (cubic,)
    linear_filtered = (linear_filtered,)
    quadratic_filtered = (quadratic_filtered,)
    cubic_filtered = (cubic_filtered,)

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)
