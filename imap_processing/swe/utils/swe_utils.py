"""Various utility classes and functions to support SWE processing."""

from enum import IntEnum

import numpy as np
import numpy.typing as npt
import pandas as pd

from imap_processing import imap_module_directory
from imap_processing.swe.utils import swe_constants


class SWEAPID(IntEnum):
    """Create ENUM for apid."""

    SWE_SCIENCE = 1344
    SWE_APP_HK = 1330
    SWE_CEM_RAW = 1334


def read_lookup_table() -> pd.DataFrame:
    """
    Read lookup table.

    Returns
    -------
    esa_table : pandas.DataFrame
        ESA table.
    """
    # Read lookup table
    lookup_table_path = imap_module_directory / "swe/l1b/swe_esa_lookup_table.csv"
    esa_table = pd.read_csv(lookup_table_path)
    return esa_table


def combine_acquisition_time(
    acq_start_coarse: np.ndarray, acq_start_fine: np.ndarray
) -> npt.NDArray:
    """
    Combine acquisition time of each quarter cycle measurement.

    Each quarter cycle data should have same acquisition start time coarse
    and fine value. We will use that as base time to calculate each
    acquisition time for each count data.
    base_quarter_cycle_acq_time = acq_start_coarse +
    |                            acq_start_fine / 1000000

    Parameters
    ----------
    acq_start_coarse : np.ndarray
        Acq start coarse. It is in seconds.
    acq_start_fine : np.ndarray
        Acq start fine. It is in microseconds.

    Returns
    -------
    np.ndarray
        Acquisition time in seconds.
    """
    epoch_center_time = acq_start_coarse + (
        acq_start_fine / swe_constants.MICROSECONDS_IN_SECOND
    )
    return epoch_center_time


def calculate_data_acquisition_time(
    acq_start_time: np.ndarray,
    esa_step_number: int,
    acq_duration: int,
    settle_duration: int,
) -> npt.NDArray:
    """
    Calculate center acquisition time of each science data point.

    Center acquisition time (in seconds) of each count data
    point at each energy and at angle step will be
    calculated using this formula:
    |    each_count_acq_time = acq_start_time +
    |           (step * ( acq_duration + settle_duration) / 1000000 )
    where 'step' goes from 0 to 179, acq_start_time is in seconds and
    settle_duration and acq_duration are in microseconds.

    To calculate center time of data acquisition time, we will add
    |    each_count_acq_time + (acq_duration / 1000000) / 2

    Parameters
    ----------
    acq_start_time : np.ndarray
        Start acquisition time in seconds.
    esa_step_number : int
        Energy step.
    acq_duration : int
        Acquisition duration in microseconds.
    settle_duration : int
        Settle duration in microseconds.

    Returns
    -------
    esa_step_number_acq_time : np.ndarray
        ESA step number acquisition center time in seconds.
    """
    # Calculate time for each ESA step
    esa_step_number_acq_time = (
        acq_start_time
        + (
            esa_step_number
            * (acq_duration + settle_duration)
            / swe_constants.MICROSECONDS_IN_SECOND
        )
        + (acq_duration / swe_constants.MICROSECONDS_IN_SECOND) / 2
    )

    return esa_step_number_acq_time
