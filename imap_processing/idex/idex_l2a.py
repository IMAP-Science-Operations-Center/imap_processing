"""
Perform IDEX L2a Processing.

Examples
--------
.. code-block:: python

    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b
    from imap_processing.idex.idex_l2a import idex_l2a

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file)
    l1b_data = idex_l1b(l1a_data)
    l2a_data = idex_l2a(l1b_data)
    write_cdf(l2a_data)
"""

import logging
from enum import IntEnum

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import butter, detrend, filtfilt, find_peaks
from scipy.stats import exponnorm

from imap_processing import imap_module_directory
from imap_processing.idex import idex_constants
from imap_processing.idex.idex_constants import SPICE_ARRAYS
from imap_processing.idex.idex_utils import get_idex_attrs, setup_dataset

logger = logging.getLogger(__name__)


class BaselineNoiseTime(IntEnum):
    """
    Time range in microseconds that mark the baseline noise before a Dust impact.

    Attributes
    ----------
    START: int
         Beginning of the baseline noise window.
    STOP: int
        End of the baseline noise window.
    """

    START = -7
    STOP = -5


def idex_l2a(l1b_dataset: xr.Dataset) -> xr.Dataset:
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

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    # Get attributes
    idex_attrs = get_idex_attrs("l2a")
    logger.info(
        f"Running IDEX L2A processing on dataset: {l1b_dataset.attrs['Logical_source']}"
    )

    tof_high = l1b_dataset["TOF_High"]
    hs_time = l1b_dataset["time_high_sample_rate"]
    ls_time = l1b_dataset["time_low_sample_rate"]

    # Load an array of known masses of ions
    atomic_masses_path = f"{imap_module_directory}/idex/atomic_masses.csv"
    atomic_masses = pd.read_csv(atomic_masses_path)
    masses = atomic_masses["Mass"]
    stretches, shifts, mass_scales = time_to_mass(tof_high.data, hs_time.data, masses)

    # TODO use correct fillval
    mass_scales_da = xr.DataArray(
        name="mass_scale",
        data=mass_scales,
        dims=("epoch", "time_high_sample_rate_index"),
        attrs=idex_attrs.get_variable_attributes("mass_scale"),
    )
    snr = calculate_snr(tof_high, hs_time)
    # Find peaks for each event. The peaks represent a TOF of an ion.
    # Peaks_2d is a list of variable-length arrays
    peaks_2d = [find_peaks(tof, prominence=0.01)[0] for tof in tof_high]
    kappa = calculate_kappa(mass_scales, peaks_2d)

    # Analyze peaks for estimating dust composition
    peak_fits_params, area_under_fits, fit_chisqr, fit_redchi = xr.apply_ufunc(
        analyze_peaks,
        tof_high,
        hs_time,
        mass_scales_da,
        np.arange(len(peaks_2d)),
        kwargs={"peaks_2d": peaks_2d},
        input_core_dims=[
            ["time_high_sample_rate_index"],
            ["time_high_sample_rate_index"],
            ["time_high_sample_rate_index"],
            [],
        ],
        output_core_dims=[
            ["mass_index", "peak_fit_parameters_index"],
            ["mass_index"],
            ["mass_index"],
            ["mass_index"],
        ],
        vectorize=True,
        keep_attrs=True,
    )
    peak_fits_params.attrs = idex_attrs.get_variable_attributes(
        "tof_peak_fit_parameters"
    )
    area_under_fits.attrs = idex_attrs.get_variable_attributes(
        "tof_peak_area_under_fit"
    )
    fit_chisqr.attrs = idex_attrs.get_variable_attributes("tof_peak_chi_squared")
    fit_redchi.attrs = idex_attrs.get_variable_attributes(
        "tof_peak_reduced_chi_squared"
    )

    area_under_fits.rename("tof_peak_area_under_fit")
    peak_fits_params.rename("tof_peak_fit_parameters")
    fit_chisqr.rename("tof_peak_chi_squared")
    fit_redchi.rename("tof_peak_reduced_chi_squared")
    # Create l2a Dataset
    prefixes = ["time_low_sample", "time_high_sample"]
    data_vars = {
        "tof_peak_fit_parameters": peak_fits_params,
        "tof_peak_area_under_fit": area_under_fits,
        "mass_scale": mass_scales_da,
        "tof_peak_chi_square": fit_chisqr,
        "tof_peak_reduced_chi_square": fit_redchi,
    }

    l2a_dataset = setup_dataset(
        l1b_dataset, prefixes + SPICE_ARRAYS, idex_attrs, data_vars
    )

    for waveform in ["Target_Low", "Target_High", "Ion_Grid"]:
        # Get the dust mass estimates and fit results
        fit_results = xr.apply_ufunc(
            estimate_dust_mass,
            ls_time,
            l1b_dataset[waveform],
            input_core_dims=[
                ["time_low_sample_rate_index"],
                ["time_low_sample_rate_index"],
            ],
            output_core_dims=[
                ["target_fit_parameter_index"],
                [],
                [],
                [],
                ["time_low_sample_rate_index"],
            ],
            vectorize=True,
            output_dtypes=[np.float64] * 6,
            keep_attrs=True,
        )
        waveform_name = waveform.lower()
        output_vars = {
            f"{waveform_name}_fit_parameters": fit_results[0],
            f"{waveform_name}_impact_charge": fit_results[1],
            f"{waveform_name}_dust_mass_estimate": fit_results[1],
            # Same as impact_charge for now
            f"{waveform_name}_chi_squared": fit_results[2],
            f"{waveform_name}_reduced_chi_squared": fit_results[3],
            f"{waveform_name}_fit_results": fit_results[4],
        }
        # Add variables to dataset with attrs
        for name, data in output_vars.items():
            l2a_dataset[name] = data
            l2a_dataset[name].attrs = idex_attrs.get_variable_attributes(name)

    l2a_dataset["mass"] = mass_scales_da

    l2a_dataset["tof_peak_kappa"] = xr.DataArray(
        kappa,
        dims="epoch",
        attrs=idex_attrs.get_variable_attributes("tof_peak_kappa"),
    )
    l2a_dataset["tof_snr"] = xr.DataArray(
        snr,
        dims="epoch",
        attrs=idex_attrs.get_variable_attributes("tof_snr"),
    )
    # Add index and label arrays
    l2a_dataset["mass_index"] = xr.DataArray(
        name="mass_index",
        data=np.arange(len(area_under_fits[0])),
        dims="mass_index",
        attrs=idex_attrs.get_variable_attributes("mass_index", check_schema=False),
    )
    l2a_dataset["peak_fit_parameter_index"] = xr.DataArray(
        name="peak_fit_parameter_index",
        data=np.arange(len(l2a_dataset["tof_peak_fit_parameters"][0][0])),
        dims="peak_fit_parameter_index",
        attrs=idex_attrs.get_variable_attributes(
            "peak_fit_parameter_index", check_schema=False
        ),
    )
    l2a_dataset["target_fit_parameter_index"] = xr.DataArray(
        name="target_fit_parameter_index",
        data=np.arange(5),
        dims="target_fit_parameter_index",
        attrs=idex_attrs.get_variable_attributes(
            "target_fit_parameter_index", check_schema=False
        ),
    )
    l2a_dataset["mass_labels"] = xr.DataArray(
        name="mass_labels",
        data=l2a_dataset.mass_index.astype(str),
        dims="mass_index",
        attrs=idex_attrs.get_variable_attributes("mass_labels", check_schema=False),
    )
    l2a_dataset["peak_fit_parameter_labels"] = xr.DataArray(
        name="peak_fit_parameter_labels",
        data=np.array(["mu", "sigma", "lambda"]),
        dims="peak_fit_parameter_index",
        attrs=idex_attrs.get_variable_attributes(
            "peak_fit_parameter_labels", check_schema=False
        ),
    )
    l2a_dataset["target_fit_parameter_labels"] = xr.DataArray(
        name="target_fit_parameter_labels",
        data=np.array(
            [
                "time_of_impact",
                "constant_offset",
                "amplitude",
                "rise_time",
                "discharge_time",
            ]
        ),
        dims="target_fit_parameter_index",
        attrs=idex_attrs.get_variable_attributes(
            "target_fit_parameter_labels", check_schema=False
        ),
    )
    logger.info("IDEX L2A science data processing completed.")
    l2a_dataset.attrs.update(idex_attrs.get_global_attributes("imap_idex_l2a_sci"))
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
        The time of flight array for one dust event. Shape is
        (epoch, high_time_sample_rate).
    high_sampling_time : numpy.ndarray
        The high sampling time array for one dust event. Shape is
        (epoch, high_time_sample_rate).
    masses : np.ndarray
        Array of known masses of ions. Shape is (21,).

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

    # Normalize time so start time is zero.
    # This is necessary to find the correct time offset
    time = high_sampling_time - high_sampling_time[:, 0:1]

    # Start with a time offset of 0
    t_offset = 0
    shift = np.zeros((len(random_stretches), len(tof_high)))
    correlation = np.zeros_like(shift)
    # Step 1
    t_i = np.zeros((len(random_stretches), len(tof_high[0])))
    # Step 2
    t_calc = t_offset + random_stretches[:, np.newaxis] * np.sqrt(np.array(masses))
    for i in range(len(random_stretches)):
        # Round every calculated time to the nearest int
        t_calc_int = np.round(t_calc[i]).astype(int)
        # Set values of t_i to 1 at the rounded calculated times if the time is less
        # than the length of t_i
        # E.g., if t_calc_int[0] = 5 then t_i[5] = 1.
        t_i[i, t_calc_int[t_calc_int < len(t_i[0])]] = 1
        # Step 3
        # Cross-correlate t_i with TOF
        # T_i simulates peaks at the times expected from the formula above,
        # when this is cross correlated with the actual time of flight array with
        # The measured peaks, we can measure the lags between them.
        for j in range(len(tof_high)):
            cross_correlation = np.correlate(t_i[i], tof_high[j], mode="full")
            if np.all(cross_correlation == 0):
                logger.warning(
                    "There are no correlations found between the TOF array "
                    "and the expected mass times array. The resulting mass scale "
                    "may be inaccurate."
                )
            # Find the lag corresponding to the maximum correlation
            # Represents the time lag from where the arrays are most correlated

            # When np.correlate mode is 'full', it returns the convolution at each
            # point of overlap, with an output shape of (N+M-1,) where N and M are the
            # lengths of the input arrays. The center point or zero lag is at index
            # len(M) - 1. Positions before this are negative lags, and
            # positions after are positive lags.
            middle = len(t_i[0]) - 1
            shift[i, j] = np.argmax(cross_correlation) - middle
            correlation[i, j] = np.max(cross_correlation)

    # Calculate the estimated mass for each time (after the time has been aligned using
    # the best t_offset and stretch_factor and converted to seconds).
    # Step 4
    # Gets the best shift in seconds (shift is currently in number of samples)
    best_shift = (
        idex_constants.FM_SAMPLING_RATE
        * shift[np.argmax(correlation, axis=0), np.arange(len(shift[0]))]
    )
    # Get the best stretch in seconds
    best_stretch = (
        idex_constants.NS_TO_S * random_stretches[np.argmax(correlation, axis=0)]
    )

    mass_scale = (
        (time * idex_constants.US_TO_S - best_shift[:, np.newaxis])
        / best_stretch[:, np.newaxis]
    ) ** 2

    return best_stretch, best_shift, mass_scale


def calculate_kappa(mass_scales: np.ndarray, peaks_2d: list) -> NDArray:
    """
    Calculate the kappa value for each mass scale.

    Kappa represents the difference between the observed mass peaks and their
    expected integer values in the calculated mass scale. The value ranges between zero
    and one. A kappa value closer to zero indicates a better accuracy of the mass scale.

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
        tof_high,
        np.nan,
    )
    if np.all(np.isnan(baseline_noise)):
        logger.warning(
            "Unable to find baseline noise. "
            f"There is no signal from {BaselineNoiseTime.START} to "
            f"{BaselineNoiseTime.STOP} ns. Returning np.nan SNR values"
        )
        return np.full(len(hs_time), fill_value=np.nan)
    # Get the max signal without baseline noise
    tof_max = np.max(tof_high, axis=1) - np.nanmean(baseline_noise, axis=1)
    tof_sigma = np.nanstd(baseline_noise, axis=1, ddof=1)
    # Return snr ratio
    return tof_max / tof_sigma


def analyze_peaks(
    tof_high: xr.DataArray,
    high_sampling_time: xr.DataArray,
    mass_scale: xr.DataArray,
    event_num: int,
    peaks_2d: np.ndarray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Fit an EMG curve to the Time of Flight data around each peak.

    Parameters
    ----------
    tof_high : xarray.DataArray
        The time of flight array.
    high_sampling_time : xarray.DataArray
        The high sampling time array.
    mass_scale : xarray.DataArray
        Time to mass scale.
    event_num : int
        Dust event number (for debugging purposes).
    peaks_2d : numpy.ndarray
        Nested list of peak indices.

    Returns
    -------
    params: numpy.ndarray
        Array of the EMG fit parameters (mu, sigma, lambda) at the corresponding mass.
        Empty mass slots contain zeros.

    area_under_emg : numpy.ndarray
        Array of the area under the EMG curve at that mass. Empty mass slots
        contain zeros.
    """
    # Initialize arrays to store EMG fit results
    # fit_params: (500, 3) array where the first dimension is the estimated ion mass (
    # 0-499)
    # and the second is EMG fit parameters (mu, sigma, lambda) for peaks at that mass
    # area_under_emg: (500) array storing the area under each EMG peak at
    # corresponding mass.
    ion_mass_dim = 500
    fit_params = np.zeros((ion_mass_dim, 3))
    area_under_emg = np.zeros(ion_mass_dim)
    chisqrs = np.zeros(ion_mass_dim)
    redchis = np.zeros(ion_mass_dim)
    for peak in peaks_2d[event_num]:
        # Take a slice of 5 samples on either side of the peak
        start = max(0, peak - 5)
        end = min(len(tof_high), peak + 6)

        time_slice = high_sampling_time[start:end]
        tof_slice = tof_high[start:end]

        param, chisqr, redchi = fit_emg(time_slice, tof_slice, event_num)
        if np.all(np.isnan(param)):
            continue

        area = calculate_area_under_emg(time_slice, param)
        # extract the variables
        k, mu, sigma = param
        # Calculate lambda
        lam = 1 / (k * sigma)
        # Find the index where time is closest to mu
        time_idx = np.argmin(np.abs(high_sampling_time.data - mu))
        mass = mass_scale[time_idx]
        # Round calculated mass to get the index
        # If that index is already taken, keep increasing the index by one
        # until we find an empty slot.
        # This ensures we don't overwrite existing data when we have multiple peaks
        # close to the same mass number
        if mass < 0:
            logger.warning(f"Warning: Calculated a negative mass: {mass}.")

        mass = max(0, round(mass))
        # Find the first index with non-zero fit parameters, starting from current mass
        non_zero_idxs = np.nonzero(np.all(fit_params[mass:] != 0, axis=-1))[0]
        # Determine index to use
        # If no non-zero parameters found, use current mass index
        # Otherwise, use the current mass plus offset to first non-zero index
        idx = mass if not non_zero_idxs.size else mass + non_zero_idxs[0]

        if idx < 500:
            fit_params[idx] = np.array([mu, sigma, lam])
            area_under_emg[idx] = area
            chisqrs[idx] = chisqr
            redchis[idx] = redchi
        else:
            logger.warning(f"Unable to find a slot for mass: {mass}. Discarding value.")

    return fit_params, area_under_emg, chisqrs, redchis


def fit_emg(
    peak_time: np.ndarray, peak_signal: np.ndarray, event_num: int
) -> tuple[NDArray, float, float]:
    """
    Fit an exponentially modified gaussian function to the peak signal.

    Scipy.stats.exponnorm.pdf uses parameters shape (k),
    location (mu), and scale (sigma) where k = 1/(sigma*lambda)
    with lambda being the exponential decay rate.

    Parameters
    ----------
    peak_time : numpy.ndarray
        TOF high +5 and -5 samples around peak.
    peak_signal : numpy.ndarray
        High sampling time array at +5 and -5 samples around peak.
    event_num : int
        Dust event number (for debugging purposes).

    Returns
    -------
    param : numpy.ndarray
        Fitted EMG optimal values for the parameters (popt) [k (shape parameter), mu,
        sigma] if fit is successful, array of np.nans otherwise.
    chisqr : float
        Chi-square value if fit is successful, np.nan otherwise.
    redchi : float
        Reduced chi-square value if fit is successful, np.nan otherwise.
    """
    # Initial Guess for the parameters of the emg fit:
    # center of gaussian
    mu = peak_time[np.argmax(peak_signal)]
    sigma = np.std(peak_time) / 10
    # Decay rate
    lam = 1 / (peak_time[-1] - peak_time[0])
    # Calculate shape parameter K from lambda and sigma
    k = 1 / (lam * sigma)
    p0 = [k, mu, sigma]

    try:
        param, _ = curve_fit(
            exponnorm.pdf, peak_time, peak_signal, p0=p0, maxfev=100_000
        )

    except RuntimeError as e:
        logger.warning(
            f"Failed to fit EMG curve: {e}\n"
            f"Time range: {peak_time[0]:.2f} to {peak_time[-1]:.2f}\n"
            f"Signal range: {min(peak_signal):.2f} to {max(peak_signal):.2f}\n"
            f"Event number: {event_num}\n"
            "Returning np.nan values."
        )
        return np.full(len(p0), np.nan), np.nan, np.nan

    emg_fit = exponnorm.pdf(peak_time, *param)
    chisqr, redchi = chi_square(peak_signal, emg_fit, len(p0))

    return param, chisqr, redchi


def calculate_area_under_emg(time_slice: np.ndarray, param: np.ndarray) -> float:
    """
    Calculate the area under the emg fit which is equal to the impact charge.

    Parameters
    ----------
    time_slice : numpy.ndarray
        Time values around the peak.
    param : numpy.ndarray
        Optimal parameters (k, mu, sigma) for the emg curve fit.

    Returns
    -------
    float
        Total area under the emg curve.
    """
    # Extract EMG fit parameters: k, mu, sigma
    k, mu, sigma = param
    # Compute integral
    area, _ = quad(exponnorm.pdf, time_slice[0], time_slice[-1], args=(k, mu, sigma))

    return float(area)


def estimate_dust_mass(
    low_sampling_time: xr.DataArray,
    target_signal: xr.DataArray,
    remove_noise: bool = True,
) -> tuple[NDArray, float, float, float, NDArray]:
    """
    Filter and fit the target or ion grid signals to get the total dust impact charge.

    Parameters
    ----------
    low_sampling_time : xarray.DataArray
        The low sampling time array.
    target_signal : xarray.DataArray
        Target signal data.
    remove_noise : bool
        If true, attempt to remove background noise, otherwise fit on the unfiltered
        signal.

    Returns
    -------
    param : numpy.ndarray
        Optimal target signal fit values for the parameters (popt)
        [time_of_impact, constant_offset, amplitude, rise_time, discharge_time]
        if fit successful. None otherwise.
    sig_amp : float
        Signal amplitude, calculated as difference between fitted maximum signal
        and baseline mean if fit successful. None otherwise.
    chi_squared : float
        Sum of squared residuals from the fit.
    reduced_chi_squared : float
        Chi-squared per degree of freedom.
    result : numpy.ndarray
        The model values evaluated at each time point.
    """
    signal = np.array(target_signal.data)
    time = np.array(low_sampling_time.data)
    good_mask = np.logical_and(
        time >= BaselineNoiseTime.START,
        time <= BaselineNoiseTime.STOP,
    )
    if not np.any(good_mask):
        logger.warning(
            "Unable to find baseline noise. "
            f"There is no signal from {BaselineNoiseTime.START} to "
            f"{BaselineNoiseTime.STOP} ns."
        )
    if remove_noise:
        # Remove noise due to "microphonics"
        signal = remove_signal_noise(time, signal, good_mask)
    # Time before image charge
    pre = -2.0
    # Get signal values where the time is before the image charge
    signal_before_imapact = signal[time < pre]
    # Center the baseline signal around zero
    signal_baseline = signal_before_imapact - np.mean(signal_before_imapact)

    # Initial Guess for the parameters of the ion grid signal
    time_of_impact = 0.0  # Time of dust hit
    constant_offset = 0.0  # Initial baseline
    amplitude: float = np.max(signal)  # Signal height
    rise_time = 0.371  # How fast the signal rises (s)
    discharge_time = 0.371  # How fast signal decays (s)

    p0 = [time_of_impact, constant_offset, amplitude, rise_time, discharge_time]

    try:
        with np.errstate(invalid="ignore", over="ignore"):
            param, _ = curve_fit(
                fit_impact,
                time,
                signal,
                p0=p0,
                maxfev=100_000,  # , epsfcn=1e-10
            )
    except RuntimeError as e:
        logger.warning(
            f"Failed to fit curve: {e}\n"
            f"Time range: {time[0]:.2f} to {time[-1]:.2f}\n"
            f"Signal range: {min(signal):.2f} to {max(signal):.2f}\n"
            "Returning None."
        )
        return (
            np.full(len(p0), np.nan),
            np.nan,
            np.nan,
            np.nan,
            np.full_like(time, np.nan),
        )

    impact_fit = fit_impact(time, *param)
    # Calculate the resulting signal amplitude after removing baseline noise
    sig_amp = max(impact_fit) - np.mean(signal_baseline)
    chisqr, redchi = chi_square(signal, impact_fit, len(p0))

    return param, float(sig_amp), chisqr, redchi, impact_fit


def fit_impact(
    time: np.ndarray,
    time_of_impact: float,
    constant_offset: float,
    amplitude: float,
    rise_time: float,
    discharge_time: float,
) -> NDArray:
    """
    Fit function for the Ion Grid and two target signals.

    Parameters
    ----------
    time : np.ndarray
        Time values for the signal.
    time_of_impact : float
        Time of dust impact.
    constant_offset : float
        Initial baseline noise.
    amplitude : float
        Signal height.
    rise_time : float
        How fast the signal rises (s).
    discharge_time : float
        How fast the signal decays (s).

    Returns
    -------
    np.ndarray
        Function values calculated at the input time points.

    Notes
    -----
    Impact charge fit function [1]_:
    Y(t) = C₀ + H(t - t₀)[C₂(1 - e^(-(t-t₀)/τ₁))e^(-(t-t₀)/τ₂) - C₁]

    References
    ----------
    .. [1] Horányi, M., et al. (2014), The Lunar Dust Experiment (LDEX) Onboard the
       Lunar Atmosphere and Dust Environment Explorer (LADEE) mission, Space Sci. Rev.,
       185(1–4), 93–113, doi:10.1007/s11214-014-0118-7.
    """
    exponent_1 = 1.0 - np.exp(-(time - time_of_impact) / rise_time)
    exponent_2 = np.exp(-(time - time_of_impact) / discharge_time)
    return constant_offset + np.heaviside(time - time_of_impact, 0) * (
        amplitude * exponent_1 * exponent_2
    )


def remove_signal_noise(
    time: np.ndarray, signal: np.ndarray, good_mask: np.ndarray
) -> NDArray:
    """
    Remove linear, sine wave, and high frequency background noise from the input signal.

    Parameters
    ----------
    time : np.ndarray
        Time values for the signal.
    signal : numpy.ndarray
        Target or Ion Grid signal.
    good_mask : numpy.ndarray
        Boolean mask for the signal array to determine where the baseline noise is.

    Returns
    -------
    numpy.ndarray
        Signal with linear, sine wave, and high frequency background noise filtered out.
    """
    # Remove linear noise
    signal = detrend(signal, type="linear")
    # Remove sine wave Background
    baseline_detrended = signal[good_mask]
    # Approximate initial values for the fit
    amplitude: float = max(baseline_detrended)
    frequency = idex_constants.TARGET_NOISE_FREQUENCY
    # Horizontal wave shift
    phase_shift = 45
    # Minimize function
    p0 = [amplitude, frequency, phase_shift]
    # Fit a sign wave to the baseline noise with initial best guesses of
    # amplitude, period, and phase shift
    try:
        # Set epsfcn to 1e-10 to mimic what lmfit minimize does
        param, _ = curve_fit(
            sine_fit,
            time[good_mask],
            baseline_detrended,
            p0=p0,
            maxfev=100_000,
            epsfcn=1e-10,
        )
        # Remove the sine wave background from the signal
        signal -= sine_fit(time, *param)
    except RuntimeError as e:
        logger.warning(f"Failed to fit background noise sine wave : {e}\n")

    # Use the butterworth filter to smooth remaining noise and remove noise above
    # desired cutoff
    signal = butter_lowpass_filter(time, signal)
    return signal


def sine_fit(time: np.ndarray, a: float, f: float, p: float) -> NDArray:
    """
    Generate a sine wave with given amplitude, angular frequency, and phase.

    Parameters
    ----------
    time : numpy.ndarray
        Time points at which to evaluate the sine wave, in seconds.
    a : float
        Amplitude of the sine wave.
    f : float
        Angular frequency of the sine wave.
    p : float
        Phase shift of the sine wave in radians.

    Returns
    -------
    numpy.ndarray
        Sine wave values calculated at the input time points.
    """
    return a * np.sin(f * time + p)


def butter_lowpass_filter(
    time: np.ndarray,
    signal: np.ndarray,
    cutoff: float = idex_constants.TARGET_HIGH_FREQUENCY_CUTOFF,
) -> NDArray:
    """
    Apply a Butterworth low-pass filter to remove high frequency noise from the signal.

    Parameters
    ----------
    time : numpy.ndarray
        Time values for the signal.
    signal : numpy.ndarray
        Target or Ion Grid signal.
    cutoff : float
        Frequency cutoff in Mhz (time is in microseconds).

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """
    sample_period = time[1] - time[0]
    # sampling frequency
    fs = (time[-1] - time[0]) / sample_period  # Hz
    # Calculate nyquist frequency
    # It is the highest frequency for the sampling frequency
    nyq = 0.5 * fs
    # sine wave can be approx represented as quadratic
    order = 2
    # Normalize the nyquist frequency. It is expected to be between 0 and 1
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, signal)
    return y


def chi_square(
    observed: np.ndarray, expected: np.ndarray, num_params: int
) -> tuple[float, float]:
    """
    Calculate the chi-square and reduced chi-square statistics.

    This implementation follows the approach used in lmfit.minimize()'s
    _calculate_statistics() method, which calculates chi-square as the sum of squared
    residuals:

        chisqr = (residual**2).sum()

    And reduced chi-square as the chi-square divided by degrees of freedom:

        ndata = len(residual)
        nfree = ndata - number_of_parameters
        redchi = chisqr / max(1, nfree)

    Parameters
    ----------
    observed : numpy.ndarray
       The observed signal.
    expected : numpy.ndarray
       The expected signal calculated with the fit parameters.
    num_params : int
       The number of parameters used in the fit.

    Returns
    -------
    chisqr : float
        The chi-square value.
    redchi : float
        The reduced chi-square value.
    """
    residuals = observed - expected
    chisqr = float(np.sum(residuals**2))
    redchi = chisqr / max(1, (len(observed) - num_params))
    return chisqr, redchi
