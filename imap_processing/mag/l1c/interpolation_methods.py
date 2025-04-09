# mypy: ignore-errors
"""Module containing interpolation methods for MAG L1C."""

import logging
from enum import Enum
from typing import Optional

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import lfilter

from imap_processing.mag.constants import POSSIBLE_RATES, VecSec

logger = logging.getLogger(__name__)


def remove_invalid_output_timestamps(
    input_timestamps: np.ndarray, output_timestamps: np.ndarray
) -> np.ndarray:
    """
    Remove output timestamps where we don't have input timestamps to interpolate.

    I.E. We should never create science data outside of the timeline of burst data.

    Parameters
    ----------
    input_timestamps : numpy.ndarray
        List of input timestamps (from burst data).
    output_timestamps : numpy.ndarray
        List of output timestamps (from norm data) to downsample/interpolate to.

    Returns
    -------
    numpy.ndarray
        All valid output timestamps where there exists input data.
    """
    if input_timestamps[0] > output_timestamps[0]:
        # Chop data where we don't have input timestamps to interpolate
        output_timestamps = output_timestamps[output_timestamps >= input_timestamps[0]]
    return output_timestamps


def linear(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
    input_rate: Optional[VecSec] = None,
    output_rate: Optional[VecSec] = None,
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
    input_rate : VecSec, optional
        Not required for this interpolation method.
    output_rate : VecSec, optional
        Not required for this interpolation method.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    # TODO: Remove invalid timestamps using remove_invalid_output_timestamps
    spline = make_interp_spline(input_timestamps, input_vectors, k=1)
    return spline(output_timestamps)


def quadratic(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
    input_rate: Optional[VecSec] = None,
    output_rate: Optional[VecSec] = None,
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
    input_rate : VecSec, optional
        Not required for this interpolation method.
    output_rate : VecSec, optional
        Not required for this interpolation method.

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
    input_rate: Optional[VecSec] = None,
    output_rate: Optional[VecSec] = None,
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
    input_rate : VecSec, optional
        Not required for this interpolation method.
    output_rate : VecSec, optional
        Not required for this interpolation method.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    spline = make_interp_spline(input_timestamps, input_vectors, k=3)
    return spline(output_timestamps)


def estimate_rate(timestamps: np.ndarray) -> VecSec:
    """
    Given a set of timestamps, estimate the rate of the timestamps.

    This rate will be one of the defined rates in the VecSec enum. The calculation
    assumes there are no significant gaps in the timestamps.

    Parameters
    ----------
    timestamps : numpy.ndarray
        1D array of timestamps to estimate the rate of.

    Returns
    -------
    VecSec
        Estimated rate of the timestamps.
    """
    samples_per_second = timestamps.shape[0] / (timestamps[-1] - timestamps[0]) * 1e9
    per_second = VecSec(
        POSSIBLE_RATES[(np.abs(POSSIBLE_RATES - samples_per_second)).argmin()]
    )

    return per_second


def cic_filter(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
    input_rate: Optional[VecSec],
    output_rate: Optional[VecSec],
):
    """
    Apply CIC filter to data before interpolating.

    The filtering uses a Cascaded integrator-comb (CIC) filter which is used in FSW to
    filter down the raw data to telemetered data.

    This assumes that the input_vectors and input_timestamps are downsampled to
    the output_timestamps rate. Neither input_timestamps nor output_timestamps should
    have significant gaps.

    After the CIC filter is applied, the timestamps at the beginning and end of the
    output are invalid. Therefore, we must pass in extra values and remove them after
    applying the filter. This needs about double the input values to work.

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
    input_rate : VecSec, optional
        Expected rate of input timestamps.
    output_rate : VecSec, optional
        Expected rate of output timestamps.

    Returns
    -------
    input_filtered, vectors_filtered : tuple[numpy.ndarray]
        Filtered input timestamps and filtered input vectors.
    """
    # output rate should always be higher
    input_rate = estimate_rate(input_timestamps) if input_rate is None else input_rate
    output_rate = (
        estimate_rate(output_timestamps) if output_rate is None else output_rate
    )

    if input_rate.value <= output_rate.value:
        raise ValueError(
            f"Burst mode input rate {input_rate} should never be less than "
            f"the normal mode output rate {output_rate}. "
            f"Both rates are required"
        )

    decimation_factor = int(input_rate.value / output_rate.value)
    cic1 = np.ones(decimation_factor)
    cic1 = cic1 / decimation_factor
    cic2 = np.convolve(cic1, cic1)
    delay = (len(cic2) - 1) // 2
    input_filtered = input_timestamps
    if delay != 0:
        input_filtered = input_timestamps[:-delay]

    vectors_filtered = lfilter(cic2, 1, input_vectors, axis=0)[delay:]
    return input_filtered, vectors_filtered


def linear_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
    input_rate: Optional[VecSec] = None,
    output_rate: Optional[VecSec] = None,
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
    input_rate : VecSec, optional
        Expected rate of input timestamps to be passed into the CIC filter. If not
        provided, this will be estimated.
    output_rate : VecSec, optional
        Expected rate of output timestamps to be passed into the CIC filter. If not
        provided, this will be estimated.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    input_filtered, vectors_filtered = cic_filter(
        input_vectors, input_timestamps, output_timestamps, input_rate, output_rate
    )
    return linear(vectors_filtered, input_filtered, output_timestamps)


def quadratic_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
    input_rate: Optional[VecSec] = None,
    output_rate: Optional[VecSec] = None,
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
    input_rate : VecSec, optional
        Expected rate of input timestamps to be passed into the CIC filter. If not
        provided, this will be estimated.
    output_rate : VecSec, optional
        Expected rate of output timestamps to be passed into the CIC filter. If not
        provided, this will be estimated.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    input_filtered, vectors_filtered = cic_filter(
        input_vectors, input_timestamps, output_timestamps, input_rate, output_rate
    )
    return quadratic(vectors_filtered, input_filtered, output_timestamps)


def cubic_filtered(
    input_vectors: np.ndarray,
    input_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
    input_rate: Optional[VecSec] = None,
    output_rate: Optional[VecSec] = None,
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
    input_rate : VecSec, optional
        Expected rate of input timestamps to be passed into the CIC filter. If not
        provided, this will be estimated.
    output_rate : VecSec, optional
        Expected rate of output timestamps to be passed into the CIC filter. If not
        provided, this will be estimated.

    Returns
    -------
    numpy.ndarray
        Interpolated vectors of shape (m, 3) where m is equal to the number of output
        timestamps. Contains x, y, z components of the vector.
    """
    input_filtered, vectors_filtered = cic_filter(
        input_vectors, input_timestamps, output_timestamps, input_rate, output_rate
    )
    return cubic(vectors_filtered, input_filtered, output_timestamps)


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
