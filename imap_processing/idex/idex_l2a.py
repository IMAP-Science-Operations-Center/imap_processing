"""
Perform IDEX L2a Processing.

Examples
--------
.. code-block:: python

    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b
    from imap_processing.idex.idex_l2a import idex_l2a

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file, data_version)
    l1b_data = idex_l1b(l1a_data, data_version)
    l2a_data = idex_l2a(l1b_data, data_version)
    write_cdf(l2a_data)
"""

# ruff: noqa: PLR0913
import logging
from enum import IntEnum

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy.signal import find_peaks

from imap_processing import imap_module_directory
from imap_processing.idex import idex_constants
from imap_processing.idex.idex_l1a import get_idex_attrs

logger = logging.getLogger(__name__)


class BaselineNoiseTime(IntEnum):
    """
    Time range in nanoseconds that mark the baseline noise before a Dust impact.

    Attributes
    ----------
    STOP: int
         Beginning of the baseline noise window.
    START: int
        End of the baseline noise window.
    """

    STOP = -5
    START = -7


def idex_l2a(l1b_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process IDEX l1b data to create l2a data products.

    This will use fits to estimate the total impact charge for the Ion Grid and two
    target signals.

    Calculate mass scales for each event using the TOF high arrays (best quality of the
    3 gain stages).
    The TOF peaks are fitted to EMG curves to determine total intensity, max amplitude,
    and signal quality.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        IDEX L1a dataset to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L2A processing on dataset: {l1b_dataset.attrs['Logical_source']}"
    )

    tof_high = l1b_dataset["TOF_High"]
    hs_time = l1b_dataset["time_high_sr"]

    # Load an array of known masses of ions
    atomic_masses_path = f"{imap_module_directory}/idex/atomic_masses.csv"
    atomic_masses = pd.read_csv(atomic_masses_path)
    masses = np.round(atomic_masses["Mass"], 1)
    stretches, shifts, mass_scales = time_to_mass(tof_high.data, hs_time.data, masses)

    mass_scales_da = xr.DataArray(
        name="mass_scale",
        data=mass_scales,
        dims=("epoch", "time_high_sr_dim"),
    )
    snr = calculate_snr(tof_high, hs_time)
    # Find peaks for each event. The peaks represent a TOF of an ion.
    # Peaks_2d is a list of variable-length arrays
    peaks_2d = [find_peaks(tof, prominence=0.01)[0] for tof in tof_high]
    kappa = calculate_kappa(mass_scales, peaks_2d)

    l2a_dataset = l1b_dataset.copy()

    l2a_dataset["tof_peak_kappa"] = xr.DataArray(kappa, dims=["epoch"])
    l2a_dataset["tof_snr"] = xr.DataArray(snr, dims=["epoch"])
    l2a_dataset["mass"] = mass_scales_da
    # Update global attributes
    idex_attrs = get_idex_attrs(data_version)
    l2a_dataset.attrs = idex_attrs.get_global_attributes("imap_idex_l2a_sci")

    logger.info("IDEX L2A science data processing completed.")
    return l2a_dataset


def time_to_mass(
    tof_high: np.ndarray, high_sampling_time: np.ndarray, masses: np.ndarray
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Calculate a mass scale for each TOF array in 'TOF_high'.

    1) Make a vector with all zeros and a length of 8189, same as the TOF length: t_i
    2) Calculate the times when each input mass should appear in the TOF data: t_calc
        for each mass, calculate a time using this formula:

            t_calc = t_offset + stretch_factor*sqrt(mass)

            t_offset is the time offset (ns)
            stretch factor (ns)

        Then and set the value at the index of t_i that is closest to each of the
        t_calcs to 1, the rest stay zero.
    3) Calculate the cross-correlation with the original TOF.
        The max will give you the best lag (t_offset) for a given stretch_factor.
    4) Choose the stretch_factor that has the highest correlation

    Parameters
    ----------
    tof_high : numpy.ndarray
        The time of flight array for one dust event.
    high_sampling_time : numpy.ndarray
        The high sampling time array for one dust event.
    masses : np.ndarray
        Array of known masses of ions.

    Returns
    -------
    numpy.ndarray
        Best stretch value per event(adjusts scale).
    numpy.ndarray
        Best shift value per event (shifts scale left or right).
    numpy.ndarray
        Estimated mass for each time per event (after the time has been aligned using
        the best t_offset and stretch_factor).
    """
    # Create an array of random stretches
    # eventually, the stretch_factor used to create the highest correlation is used to
    # align the time
    min_stretch = 1400
    random_stretches = np.linspace(min_stretch, min_stretch + 100, 10)

    time = high_sampling_time - high_sampling_time[:, 0:1]

    # Start with a time offset of 0
    t_offset = 0
    shift = np.zeros((len(random_stretches), len(tof_high)))
    corr = np.zeros_like(shift)
    for i in range(len(random_stretches)):
        # Step 1
        t_i = np.zeros(len(tof_high[0]))
        # Step 2
        t_calc = t_offset + random_stretches[i] * np.sqrt(masses)
        # Loop through the elements of t_calc and set corresponding elements in t_i to 1
        for idx in np.round(t_calc).astype(int):
            if 0 <= idx < len(t_i):
                t_i[idx] = 1
        # Step 3
        # Cross-correlate t_calc with TOF
        for j in range(len(tof_high)):
            cross_correlation = np.correlate(t_i, tof_high[j], mode="full")
            if np.all(cross_correlation == 0):
                logger.warning(
                    "There are no correlations found between the TOF array "
                    "and the expected mass times array. The resulting mass scale "
                    "may be inaccurate."
                )
            # Find the lag corresponding to the maximum correlation
            # Represents the time lag from where the arrays are most correlated
            # Zero lag position
            middle = len(t_i) - 1
            shift[i, j] = np.argmax(cross_correlation) - middle
            corr[i, j] = np.max(cross_correlation)

    # Calculate the estimated mass for each time (after the time has been aligned using
    # the best t_offset and stretch_factor and converted to seconds).
    # Step 4
    # Gets the best shift in seconds
    best_shift = (
        idex_constants.FM_SAMPLING_RATE
        * shift[np.argmax(corr, axis=0), np.arange(len(shift[0]))]
    )
    # Get the best stretch in seconds
    best_stretch = (
        idex_constants.NS_TO_S_CONV_FACTOR * random_stretches[np.argmax(corr, axis=0)]
    )

    mass_scale = (
        (time * idex_constants.MS_TO_S_CONV_FACTOR - best_shift[:, np.newaxis])
        / best_stretch[:, np.newaxis]
    ) ** 2

    return best_stretch, best_shift, mass_scale


def calculate_kappa(mass_scales: np.ndarray, peaks_2d: list) -> NDArray:
    """
    Calculate the kappa value for each peak.

    Parameters
    ----------
    mass_scales : xarray.DataArray
        Array containing the masses at each time value for each dust event.
    peaks_2d : list
        A Nested list of tof peak indices.

    Returns
    -------
    numpy.ndarray
        Average distance from the assigned peak to the nearest integer value.
    """
    #  Find the average deviation between each TOF peak's assigned mass value and its
    #  nearest decimal value per spectrum.
    kappas = np.asarray(
        [
            np.mean(mass_scale[peaks] - np.round(mass_scale[peaks]))
            for mass_scale, peaks in zip(mass_scales, peaks_2d)
        ]
    )
    return kappas


def calculate_snr(tof_high: xr.DataArray, hs_time: xr.DataArray) -> NDArray:
    """
    Calculate the signal-to-noise ratio.

    Parameters
    ----------
    tof_high : xarray.DataArray
        The time of flight array.
    hs_time : xarray.DataArray
        The high sampling time array.

    Returns
    -------
    numpy.ndarray
        Signal-to-noise ratio at each event.
    """
    # Find indices where Time (High Sampling) is between -7 and -5 ns (no signal yet)
    # To determine the baseline noise
    baseline_noise = np.where(
        np.logical_and(
            hs_time >= BaselineNoiseTime.START, hs_time <= BaselineNoiseTime.STOP
        ),
        tof_high.data,
        np.nan,
    )
    if np.all(np.isnan(baseline_noise)):
        logger.warning(
            "Unable to find baseline noise. "
            f"There is no signal from {BaselineNoiseTime.START} to "
            f"{BaselineNoiseTime.STOP} ns. Returning np.nan SNR values"
        )
        return np.zeros(len(hs_time))
    # Get the max signal without baseline noise
    tof_max = np.max(tof_high.data, axis=1) - np.nanmean(baseline_noise, axis=1)
    tof_sigma = np.nanstd(baseline_noise, axis=1, ddof=1)
    # Return snr ratio
    return tof_max / tof_sigma
