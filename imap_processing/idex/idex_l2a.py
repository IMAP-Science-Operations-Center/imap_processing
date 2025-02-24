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
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import butter, detrend, filtfilt, find_peaks
from scipy.special import erfc

from imap_processing import imap_module_directory
from imap_processing.idex import idex_constants
from imap_processing.idex.idex_constants import ConversionFactors
from imap_processing.idex.idex_l1a import get_idex_attrs

logger = logging.getLogger(__name__)


class BaselineNoiseTime(IntEnum):
    """
    Time range in microseconds that mark the baseline noise before a Dust impact.

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
    ls_time = l1b_dataset["time_low_sr"]

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

    # Analyze peaks for estimating dust composition
    peak_fits, area_under_fits = xr.apply_ufunc(
        analyze_peaks,
        tof_high,
        hs_time,
        mass_scales_da,
        np.arange(len(peaks_2d)),
        kwargs={"peaks_2d": peaks_2d},
        input_core_dims=[
            ["time_high_sr_dim"],
            ["time_high_sr_dim"],
            ["time_high_sr_dim"],
            [],
        ],
        # TODO: Determine dimension name
        output_core_dims=[
            ["time_of_flight", "peak_fit_parameters"],
            ["time_of_flight"],
        ],
        vectorize=True,
    )

    l2a_dataset = l1b_dataset.copy()

    for waveform in ["Target_Low", "Target_High", "Ion_Grid"]:
        # Convert back to raw DNs for more accurate fits
        waveform_dn = l1b_dataset[waveform] / ConversionFactors[waveform]
        # Get the dust mass estimates and fit results
        fit_results = xr.apply_ufunc(
            estimate_dust_mass,
            ls_time,
            waveform_dn,
            input_core_dims=[["time_low_sr_dim"], ["time_low_sr_dim"]],
            output_core_dims=[["fit_parameters"], [], [], [], ["time_low_sr_dim"]],
            vectorize=True,
            output_dtypes=[np.float64] * 6,
        )
        waveform_name = waveform.lower()
        # Add variables
        l2a_dataset[f"{waveform_name}_fit_parameters"] = fit_results[0]
        l2a_dataset[f"{waveform_name}_fit_imapct_charge"] = fit_results[1]
        # TODO: convert charge to mass
        l2a_dataset[f"{waveform_name}_fit_imapct_mass_estimate"] = fit_results[1]
        l2a_dataset[f"{waveform_name}_chi_squared"] = fit_results[2]
        l2a_dataset[f"{waveform_name}_reduced_chi_squared"] = fit_results[3]
        l2a_dataset[f"{waveform_name}_fit_results"] = fit_results[4]

    l2a_dataset["tof_peak_fit_parameters"] = peak_fits
    l2a_dataset["tof_peak_area_under_fit"] = area_under_fits
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


def analyze_peaks(
    tof_high: xr.DataArray,
    high_sampling_time: xr.DataArray,
    mass_scale: xr.DataArray,
    event_num: int,
    peaks_2d: np.ndarray,
) -> tuple[NDArray, NDArray]:
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
    fit_params = np.zeros((500, 3))
    area_under_emg = np.zeros(500)
    for peak in peaks_2d[event_num]:
        # Take a slice of 5 samples on either side of the peak
        start = max(0, peak - 5)
        end = min(len(tof_high), peak + 6)

        time_slice = np.asarray(high_sampling_time[start:end].data)
        tof_slice = np.asarray(tof_high[start:end].data)

        param = fit_emg(time_slice, tof_slice, event_num)
        if param is not None:
            area = calculate_area_under_emg(time_slice, param)
            # Find the index where time is closest to mu
            time_idx = np.argmin(np.abs(high_sampling_time.data - param[0]))
            mass = mass_scale[time_idx]
            # Round calculated mass to get the index
            # If that index is already taken, keep increasing the index by one
            # until we find an empty slot.
            # This ensures we don't overwrite existing data when we have multiple peaks
            # close to the same mass number
            if mass < 0:
                logger.warning(f"Warning: Calculated a negative mass: {mass}.")

            mass = max(0, round(mass))
            while np.all(fit_params[mass:] != 0) and mass < 500:
                mass += 1
            if mass < 500:
                fit_params[mass] = param
                area_under_emg[mass] = area
            else:
                logger.warning(
                    f"Unable to find a slot for mass: {mass}. Discarding " f"value."
                )

    return fit_params, area_under_emg


def fit_emg(
    peak_time: np.ndarray, peak_signal: np.ndarray, event_num: int
) -> Union[NDArray, None]:
    """
    Fit an exponentially modified gaussian function to the peak signal.

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
    param : numpy.ndarray or None
        Fitted EMG optimal values for the parameters (popt) [mu, sigma, lambda]
        if fit successful, None otherwise.
    """
    # Initial Guess for the parameters of the emg fit:
    # center of gaussian
    mu = peak_time[np.argmax(peak_signal)]
    sigma = np.std(peak_time) / 10
    # Decay rate
    lam = 1 / (peak_time[-1] - peak_time[0])

    p0 = [mu, sigma, lam]

    try:
        param, _ = curve_fit(emg, peak_time, peak_signal, p0=p0, maxfev=100_000)
    except RuntimeError as e:
        logger.warning(
            f"Failed to fit EMG curve: {e}\n"
            f"Time range: {peak_time[0]:.2f} to {peak_time[-1]:.2f}\n"
            f"Signal range: {min(peak_signal):.2f} to {max(peak_signal):.2f}\n"
            f"Event number: {event_num}\n"
            "Returning None."
        )
        return None

    return param


def emg(time: np.ndarray, mu: float, sigma: float, lam: float) -> NDArray:
    """
    Define an exponentially modified gaussian function.

    Parameters
    ----------
    time : numpy.ndarray
        Time points at which to evaluate the EMG function.
    mu : float
       The distribution mean of the gaussian.
    sigma : float
       Distribution spread about the mean.
    lam : float
        Exponential decay rate.

    Returns
    -------
    numpy.ndarray
        EMG function values calculated at the input time points.
    """
    # A normally distributed gaussian with an exponential decay.
    prefactor = lam / 2
    exponent = np.exp(prefactor * (2 * mu + lam * sigma**2 - 2 * time))
    erfc_part = erfc((mu + lam * sigma**2 - time) / (np.sqrt(2) * sigma))
    return prefactor * exponent * erfc_part


def calculate_area_under_emg(time_slice: np.ndarray, param: np.ndarray) -> float:
    """
    Calculate the area under the emg fit which is equal to the impact charge.

    Parameters
    ----------
    time_slice : numpy.ndarray
        Time values around the peak.
    param : numpy.ndarray
        Optimal parameters (mu, sigma, lam) for the emg curve fit.

    Returns
    -------
    float
        Total area under the emg curve.
    """
    # Extract EMG fit parameters: mu, sigma, lam
    mu, sigma, lam = param
    # Compute integral
    area, _ = quad(emg, time_slice[0], time_slice[-1], args=(mu, sigma, lam))

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
    # TODO: The IDEX team is iterating on this Function and will provide more
    #         information soon.
    signal = np.array(target_signal.data)
    time = np.array(low_sampling_time.data)
    mask = np.logical_and(
        time >= BaselineNoiseTime.START,
        time <= BaselineNoiseTime.STOP,
    )
    if not np.any(mask):
        logger.warning(
            "Unable to find baseline noise. "
            f"There is no signal from {BaselineNoiseTime.START} to "
            f"{BaselineNoiseTime.STOP} ns."
        )
    if remove_noise:
        # Remove noise due to "microphonics"
        signal = remove_signal_noise(time, signal, mask)
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

    # Calculate chi square and reduced chi square
    chisqr = float(np.sum((signal - impact_fit) ** 2))
    # To get reduced chi square divide by dof (number of points - number of params)
    redchi = chisqr / (len(signal) - len(p0))

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
    Fit function for the Ion Grid and two target signals given by (Horanyi, 2014).

    Y(t) = C₀ + H(t - t₀)[C₂(1 - e^(-(t-t₀)/τ₁))e^(-(t-t₀)/τ₂) - C₁]]

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
    """
    exponent_1 = 1.0 - np.exp(-(time - time_of_impact) / rise_time)
    exponent_2 = np.exp(-(time - time_of_impact) / discharge_time)
    return constant_offset + np.heaviside(time - time_of_impact, 0) * (
        amplitude * exponent_1 * exponent_2
    )


def remove_signal_noise(
    time: np.ndarray, signal: np.ndarray, mask: np.ndarray
) -> NDArray:
    """
    Remove linear, sine wave, and high frequency background noise from the input signal.

    Parameters
    ----------
    time : np.ndarray
        Time values for the signal.
    signal : numpy.ndarray
        Target or Ion Grid signal.
    mask : numpy.ndarray
        Boolean mask for the signal array to determine where the baseline noise is.

    Returns
    -------
    numpy.ndarray
        Signal with linear, sine wave, and high frequency background noise filtered out.
    """
    # Remove linear noise
    signal = detrend(signal, type="linear")
    # Remove sine wave Background
    baseline_delined = signal[mask]
    # Approximate initial values for the fit
    amplitude: float = max(baseline_delined)
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
            sine_fit, time[mask], baseline_delined, p0=p0, maxfev=100_000, epsfcn=1e-10
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
    Generate a sine wave with given amplitude, frequency, and phase.

    Parameters
    ----------
    time : numpy.ndarray
        Time points at which to evaluate the sine wave, in seconds.
    a : float
        Amplitude of the sine wave.
    f : float
        Frequency of the sine wave in Hz.
    p : float
        Phase shift of the sine wave in radians.

    Returns
    -------
    numpy.ndarray
        Sine wave values calculated at the input time points.
    """
    return a * np.sin(2 * np.pi * f * time + p)


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
    # TODO: The IDEX team might be switching this function out for a different filter.
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
