# mypy: ignore-errors
"""Module containing interpolation methods for MAG L1C."""

from enum import Enum

import numpy as np
from scipy.interpolate import make_interp_spline

from imap_processing.mag.constants import POSSIBLE_RATES


def linear(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation of input vectors to output timestamps.

    Parameters
    ----------
    input_vectors : numpy.ndarray
        Input vectors of shape (n, 3) where n is equal to the number of input
        timestamps. Contains x, y, z components of the vector.
    input_timestamps : numpy.ndarray
        Input timestamps of shape (n,) which correspond to the timestamps of the input
        vectors.
    output_timestamps : numpy.ndarray
        Output timestamps of shape (m,) to generate interpolated vectors for.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    spline = make_interp_spline(input_timestamps, input_vectors, k=1)
    return spline(output_timestamps)


def quadratic(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Quadratic interpolation of input vectors to output timestamps.

    Parameters
    ----------
    input_vectors : numpy.ndarray
        Input vectors of shape (n, 3) where n is equal to the number of input
        timestamps. Contains x, y, z components of the vector.
    input_timestamps : numpy.ndarray
        Input timestamps of shape (n,) which correspond to the timestamps of the input
        vectors.
    output_timestamps : numpy.ndarray
        Output timestamps of shape (m,) to generate interpolated vectors for.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    spline = make_interp_spline(input_timestamps, input_vectors, k=2)
    return spline(output_timestamps)


def cubic(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Cubic interpolation of input vectors to output timestamps.

    Parameters
    ----------
    input_vectors : numpy.ndarray
        Input vectors of shape (n, 3) where n is equal to the number of input
        timestamps. Contains x, y, z components of the vector.
    input_timestamps : numpy.ndarray
        Input timestamps of shape (n,) which correspond to the timestamps of the input
        vectors.
    output_timestamps : numpy.ndarray
        Output timestamps of shape (m,) to generate interpolated vectors for.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    spline = make_interp_spline(input_timestamps, input_vectors, k=3)
    return spline(output_timestamps)


def cic_filter(input_vectors, input_timestamps, output_timestamps):
    """
    Apply CIC filter to data before interpolating.

    The filtering uses a Cascaded integrator-comb (CIC) filter which is used in FSW to
    filter down the raw data to telemetered data.

    Returns
    -------

    """
    # Calculate the sample rate - count of samples / time range, then find nearest
    # possible sensor rate.
    samples_per_second = input_vectors.shape[0] / (
        input_timestamps[-1] - input_timestamps[0]
    )
    input_samples_per_sec = POSSIBLE_RATES[
        (np.abs(POSSIBLE_RATES - samples_per_second)).argmin()
    ]
    output_samples = output_timestamps.shape[0] / (
        output_timestamps[-1] - output_timestamps[0]
    )
    output_samples_per_sec = POSSIBLE_RATES[
        (np.abs(POSSIBLE_RATES - output_samples)).argmin()
    ]

    print(
        f"Input samples per second: {input_samples_per_sec}, output: {output_samples_per_sec}"
    )

    decimation_factor = input_samples_per_sec / 2


def linear_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Linear filtered interpolation of input vectors to output timestamps.

    Parameters
    ----------
    input_vectors : numpy.ndarray
        Input vectors of shape (n, 3) where n is equal to the number of input
        timestamps. Contains x, y, z components of the vector.
    input_timestamps : numpy.ndarray
        Input timestamps of shape (n,) which correspond to the timestamps of the input
        vectors.
    output_timestamps : numpy.ndarray
        Output timestamps of shape (m,) to generate interpolated vectors for.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    pass


def quadratic_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Quadratic filtered interpolation of input vectors to output timestamps.

    Parameters
    ----------
    input_vectors : numpy.ndarray
        Input vectors of shape (n, 3) where n is equal to the number of input
        timestamps. Contains x, y, z components of the vector.
    input_timestamps : numpy.ndarray
        Input timestamps of shape (n,) which correspond to the timestamps of the input
        vectors.
    output_timestamps : numpy.ndarray
        Output timestamps of shape (m,) to generate interpolated vectors for.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    pass


def cubic_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Cubic filtered interpolation of input vectors to output timestamps.

    Parameters
    ----------
    input_vectors : numpy.ndarray
        Input vectors of shape (n, 3) where n is equal to the number of input
        timestamps. Contains x, y, z components of the vector.
    input_timestamps : numpy.ndarray
        Input timestamps of shape (n,) which correspond to the timestamps of the input
        vectors.
    output_timestamps : numpy.ndarray
        Output timestamps of shape (m,) to generate interpolated vectors for.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    pass


class InterpolationFunction(Enum):
    """Enum which describes the options for interpolation functions on MAG L1C."""

    linear = (linear,)
    quadratic = (quadratic,)
    cubic = (cubic,)
    linear_filtered = (linear_filtered,)
    quadratic_filtered = (quadratic_filtered,)
    cubic_filtered = (cubic_filtered,)

    def __call__(self, *args, **kwargs):
        """
        Overwritten call which allows you to call the interpolation function directly.

        example: InterpolationFunction.linear(input_vectors, input_timestamps,
        output_timestamps)

        Parameters
        ----------
        *args : list
            List of arguments to pass to the interpolation function. Must match
            (input_vectors, input_timestamps, output_timestamps).
        **kwargs : dict
            Keyword arguments to pass to the interpolation function.

        Returns
        -------
        Any
            The return value of the interpolation function.
        """
        return self.value[0](*args, **kwargs)
