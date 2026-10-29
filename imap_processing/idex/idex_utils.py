"""Contains helper functions to support IDEX processing."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.idex.idex_constants import IDEX_10_DAY_RANGES_PATH

logger = logging.getLogger(__name__)


def calculate_baseline_and_rms(
    signal: NDArray, time: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Calculate the mean baseline and RMS noise in the first 3 microseconds.

    Parameters
    ----------
    signal : numpy.ndarray
        Waveform samples in engineering units, with samples on the last axis.
    time : numpy.ndarray
        Corresponding sample times in microseconds. Each record uses the half-open
        interval from its earliest finite time to that time plus 3 microseconds.

    Returns
    -------
    baseline : numpy.ndarray
        Mean of finite samples in the window, or NaN if none are available.
    rms : numpy.ndarray
        Root mean square deviation from the baseline (population standard
        deviation, ddof=0), in the waveform's units. NaN for an empty window.
    """
    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)
    finite_time = np.isfinite(time)
    start = np.min(
        np.where(finite_time, time, np.inf), axis=-1, keepdims=True, initial=np.inf
    )
    mask = finite_time & (time < start + 3.0) & np.isfinite(signal)
    count = np.count_nonzero(mask, axis=-1)
    baseline = np.full(signal.shape[:-1], np.nan)
    np.divide(
        np.sum(np.where(mask, signal, 0.0), axis=-1),
        count,
        out=baseline,
        where=count > 0,
    )
    residual = np.where(mask, signal - np.expand_dims(baseline, -1), 0.0)
    variance = np.full_like(baseline, np.nan)
    np.divide(np.sum(residual**2, axis=-1), count, out=variance, where=count > 0)
    if np.any(count == 0):
        logger.warning("Unable to find baseline noise in the first 3 microseconds.")
    return baseline, np.sqrt(variance)


def get_idex_attrs(data_level: str) -> ImapCdfAttributes:
    """
    Load in CDF attributes for IDEX instrument.

    Parameters
    ----------
    data_level : str
        Data level of current processing.

    Returns
    -------
    idex_attrs : ImapCdfAttributes
        The IDEX L1a CDF attributes.
    """
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs("idex")
    idex_attrs.add_instrument_variable_attrs("idex", data_level)
    return idex_attrs


def setup_dataset(
    dataset: xr.Dataset,
    match_strings: list,
    idex_attrs: ImapCdfAttributes,
    data_vars: dict | None = None,
) -> xr.Dataset:
    """
    Initialize a dataset and copy over any dataArrays.

    Parameters
    ----------
    dataset : xarray.Dataset
        Contains the arrays to copy to the new dataset. The variable named "epoch" is
        required.
    match_strings : list[str]
        Array names to copy to the new dataset.
    idex_attrs : ImapCdfAttributes
        Idex attributes for current data level processing.
    data_vars : dict
        Dictionary of variables to copy over to the new dataset.

    Returns
    -------
    new_dataset : xarray.Dataset
        Initialized dataset.
    """
    epoch_da = xr.DataArray(
        data=dataset["epoch"].data.copy(),
        name="epoch",
        dims="epoch",
        attrs=idex_attrs.get_variable_attributes("epoch", check_schema=False),
    )

    new_dataset = xr.Dataset(coords={"epoch": epoch_da}, data_vars=data_vars)

    vars_to_copy = [
        var for var in dataset.variables if any(match in var for match in match_strings)
    ]
    # Copy arrays over to the new dataset
    for var in vars_to_copy:
        new_dataset[var] = dataset[var].copy()

    return new_dataset


def get_10_day_window_end_date(start_date: str) -> str:
    """
    Use the start date to find the end date of the 10-day window.

    IDEX l1a data is processed in 10-day windows, so this function will be used to
    determine the end date of the window to process based on the start
    date passed into the job.

    Parameters
    ----------
    start_date : str
        Start date of the window to process.

    Returns
    -------
    end_date : str
        End date of the window to process.
    """
    # This CSV was provided by the IDEX team.
    idex_10_day_ranges = pd.read_csv(IDEX_10_DAY_RANGES_PATH, header=0, dtype=str)
    # Find the row where the input start date is equal to the start date in the df.
    matching_row = idex_10_day_ranges[idex_10_day_ranges["start_date"] == start_date]
    # if there is no match, raise an error. We expect that the start date passed into
    # the job will always be the start date of a 10-day window, so there should always
    # be a match in the csv.
    if matching_row.empty:
        raise ValueError(
            f"Start date {start_date} is not an IDEX defined start date"
            f" for a 10-day window."
        )
    if len(matching_row["end_date"]) > 1:
        raise ValueError(
            f"There should only be one row where start_date is equal "
            f"to {start_date}. Please check lookup table: "
            f"{IDEX_10_DAY_RANGES_PATH}."
        )
    return matching_row["end_date"].values[0]
