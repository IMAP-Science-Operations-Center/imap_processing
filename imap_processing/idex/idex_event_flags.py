"""Classify IDEX science events and identify dust-like TOF waveforms."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.signal import find_peaks

EVENT_FLAG_NAMES = (
    "science_event_flag",
    "noise_capture_flag",
    "pulser_flag",
    "dust_hit_flag",
)

SATURATION_FLAG_NAMES = (
    "tof_high_saturation_flag",
    "tof_mid_saturation_flag",
    "tof_low_saturation_flag",
    "target_high_saturation_flag",
    "target_low_saturation_flag",
    "ion_grid_saturation_flag",
)

ALL_FLAG_NAMES = EVENT_FLAG_NAMES + SATURATION_FLAG_NAMES

_TOF_MAX_DN = 1023.0
_LOW_RATE_MAX_DN = 4095.0
_SATURATION_FRACTION = 0.95
_PULSER_THRESHOLD_DN = 1000
_BASELINE_WINDOW_US = 3.0
_PEAK_THRESHOLD_SIGMA = 7.0
_PEAK_PROMINENCE_SIGMA = 5.0
_MIN_PEAK_WIDTH_US = 0.020
_SINGLE_PEAK_WIDTH_US = 0.050
_MIN_PEAK_COUNT = 2
_MIN_PEAK_DISTANCE_US = 0.030
_SATURATION_MATCH_WINDOW_US = 0.050
_TOF_REFERENCE_BASELINES = (511.0, 513.0, 514.0)
_TOF_REFERENCE_SIGMAS = (2.9652, 1.4826, 2.9652)
_TRUNCATED_BASELINE_OFFSET_DN = 20.0

_TRIGGER_CHANNELS = {
    0: "TOF H",
    1: "TOF L",
    2: "TOF M",
    3: "Target H",
}


def classify_event_flags(
    telemetry: Mapping[str, int],
    tof_high: np.ndarray,
    tof_mid: np.ndarray,
    tof_low: np.ndarray,
    time_high_sample_rate: np.ndarray,
    target_high: np.ndarray | None = None,
    target_low: np.ndarray | None = None,
    ion_grid: np.ndarray | None = None,
) -> dict[str, int]:
    """Return mutually exclusive event-type flags and the Dust Hit flag.

    The instrument state is assigned from the event trigger telemetry using
    these conditions, in order:

    * ``noise_capture_flag`` is set when no trigger channels are active, or
      when a software/external trigger is present and the only active channel
      is TOF High.
    * ``pulser_flag`` is set when TOF High is the only active channel, the TOF
      High trigger mode is ``1``, and its trigger threshold is 1000 DN.
    * ``science_event_flag`` is set for all remaining events.

    ``dust_hit_flag`` is set only for science events when the raw TOF waveform
    contains at least two peaks that exceed seven baseline-noise standard
    deviations and have a full width at half maximum of at least 20 ns. Dust
    detection uses lower-gain waveforms only to measure saturated high-gain
    peaks.

    Parameters
    ----------
    telemetry : collections.abc.Mapping
        Raw event trigger fields.
    tof_high, tof_mid, tof_low : numpy.ndarray
        Raw TOF waveforms in high, medium, and low gain.
    time_high_sample_rate : numpy.ndarray
        High-rate waveform times in microseconds.
    target_high, target_low, ion_grid : numpy.ndarray or None
        Raw low-rate waveforms used to calculate saturation flags.

    Returns
    -------
    dict[str, int]
        Event flags with values of zero or one.
    """
    trigger_id = int(telemetry.get("idx__txhdrtrigid", 0))
    active_channels = {
        channel for bit, channel in _TRIGGER_CHANNELS.items() if trigger_id & (1 << bit)
    }
    for gain, channel in (("hg", "TOF H"), ("mg", "TOF M"), ("lg", "TOF L")):
        if int(telemetry.get(f"idx__txhdr{gain}trigmode", 0)) != 0:
            active_channels.add(channel)

    has_software_or_external_trigger = bool(trigger_id & ((1 << 4) | (1 << 5)))
    hg_mode = int(telemetry.get("idx__txhdrhgtrigmode", 0))
    hg_threshold = (int(telemetry.get("idx__txhdrhgtrigctrl1", 0)) >> 22) & 0x3FF

    if not active_channels or (
        has_software_or_external_trigger and active_channels <= {"TOF H"}
    ):
        event_type = "noise_capture_flag"
    elif (
        active_channels == {"TOF H"}
        and hg_mode == 1
        and hg_threshold == _PULSER_THRESHOLD_DN
    ):
        event_type = "pulser_flag"
    else:
        event_type = "science_event_flag"

    flags = {name: 0 for name in ALL_FLAG_NAMES}
    flags[event_type] = 1
    if event_type == "science_event_flag" and _has_dust_hit(
        tof_high, tof_mid, tof_low, time_high_sample_rate
    ):
        flags["dust_hit_flag"] = 1
    flags.update(
        classify_saturation_flags(
            tof_high, tof_mid, tof_low, target_high, target_low, ion_grid
        )
    )
    return flags


def classify_saturation_flags(
    tof_high: np.ndarray,
    tof_mid: np.ndarray,
    tof_low: np.ndarray,
    target_high: np.ndarray | None,
    target_low: np.ndarray | None,
    ion_grid: np.ndarray | None,
) -> dict[str, int]:
    """Return saturation flags for the six raw waveform channels.

    The low-rate channels are optional to keep the event-classification API
    compatible with callers that only have the TOF waveforms.

    Parameters
    ----------
    tof_high, tof_mid, tof_low : numpy.ndarray
        Raw 10-bit TOF waveforms.
    target_high, target_low, ion_grid : numpy.ndarray or None
        Raw 12-bit low-rate waveforms.

    Returns
    -------
    dict[str, int]
        One zero-or-one saturation flag for each waveform channel.
    """
    waveforms = {
        "tof_high_saturation_flag": (tof_high, _TOF_MAX_DN),
        "tof_mid_saturation_flag": (tof_mid, _TOF_MAX_DN),
        "tof_low_saturation_flag": (tof_low, _TOF_MAX_DN),
        "target_high_saturation_flag": (target_high, _LOW_RATE_MAX_DN),
        "target_low_saturation_flag": (target_low, _LOW_RATE_MAX_DN),
        "ion_grid_saturation_flag": (ion_grid, _LOW_RATE_MAX_DN),
    }
    return {
        name: int(values is not None and _waveform_is_saturated(values, maximum))
        for name, (values, maximum) in waveforms.items()
    }


def _waveform_is_saturated(values: np.ndarray, maximum_dn: float) -> bool:
    """Return whether any finite waveform sample reaches the 95% limit.

    Parameters
    ----------
    values : numpy.ndarray
        Waveform samples in DN.
    maximum_dn : float
        Maximum representable DN for the channel.

    Returns
    -------
    bool
        Whether any finite sample reaches the saturation threshold.
    """
    values_array = np.asarray(values, dtype=float)
    finite_values = values_array[np.isfinite(values_array)]
    return bool(
        finite_values.size
        and np.any(finite_values >= _SATURATION_FRACTION * maximum_dn)
    )


def _has_dust_hit(
    tof_high: np.ndarray,
    tof_mid: np.ndarray,
    tof_low: np.ndarray,
    time_high_sample_rate: np.ndarray,
) -> bool:
    """Return whether any TOF gain contains a dust-like peak pattern.

    Parameters
    ----------
    tof_high, tof_mid, tof_low : numpy.ndarray
        Raw TOF waveforms in high, medium, and low gain.
    time_high_sample_rate : numpy.ndarray
        High-rate waveform times in microseconds.

    Returns
    -------
    bool
        Whether at least two peaks meet the sigma and FWHM requirements in
        one gain, or High gain contains exactly one directly measured peak
        that is at least 50 ns wide.
    """
    high = _as_1d_array(tof_high)
    mid = _as_1d_array(tof_mid)
    low = _as_1d_array(tof_low)
    times = _as_1d_array(time_high_sample_rate)
    length = min(high.size, mid.size, low.size, times.size)
    if length == 0:
        return False
    high, mid, low, times = (array[:length] for array in (high, mid, low, times))

    return _has_standard_dust_hit(high, mid, low, times) or _has_truncated_dust_hit(
        high, mid, low, times
    )


def _has_standard_dust_hit(
    high: np.ndarray, mid: np.ndarray, low: np.ndarray, times: np.ndarray
) -> bool:
    """Return whether normal baseline processing finds a dust-like pattern.

    Parameters
    ----------
    high, mid, low : numpy.ndarray
        Raw TOF waveforms in high, medium, and low gain.
    times : numpy.ndarray
        High-rate waveform times in microseconds.

    Returns
    -------
    bool
        Whether the standard peak criteria identify a dust-like pattern.
    """
    waveforms = (high, mid, low)
    peak_results = [_qualifying_peaks(waveform, times) for waveform in waveforms]
    corrected_waveforms: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        peak_results[0][0],
        peak_results[1][0],
        peak_results[2][0],
    )
    peak_sets: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        _collapse_saturated_peaks(peak_results[0][2], high, peak_results[0][0]),
        _collapse_saturated_peaks(peak_results[1][2], mid, peak_results[1][0]),
        _collapse_saturated_peaks(peak_results[2][2], low, peak_results[2][0]),
    )

    for channel_index, (_waveform, peak_result, peaks) in enumerate(
        zip(waveforms, peak_results, peak_sets, strict=True)
    ):
        corrected, sigma, _ = peak_result
        if not np.isfinite(sigma):
            continue

        widths = []
        direct_widths = []
        for peak_index in peaks:
            direct_widths.append(_fwhm(corrected, times, int(peak_index)))
            width_us = _gain_aware_width(
                channel_index,
                peak_index,
                waveforms,
                times,
                corrected_waveforms,
                peak_sets,
            )
            if np.isfinite(width_us):
                widths.append(width_us)

        # If any one gain has at least 2 peaks with FWHM >= 20 ns, it is a dust hit.
        if sum(width >= _MIN_PEAK_WIDTH_US for width in widths) >= _MIN_PEAK_COUNT:
            return True
        # If High gain has exactly one directly measured peak and it is at least
        # 50 ns wide, count it as a dust hit.
        if (
            channel_index == 0
            and len(direct_widths) == 1
            and direct_widths[0] >= _SINGLE_PEAK_WIDTH_US
        ):
            return True

    return False


def _has_truncated_dust_hit(
    high: np.ndarray, mid: np.ndarray, low: np.ndarray, times: np.ndarray
) -> bool:
    """Return whether reference-baseline processing finds two broad peaks.

    Parameters
    ----------
    high, mid, low : numpy.ndarray
        Raw TOF waveforms in high, medium, and low gain.
    times : numpy.ndarray
        High-rate waveform times in microseconds.

    Returns
    -------
    bool
        Whether the fallback peak criteria identify two qualifying peaks.
    """
    # If the impact begins before the TOF record, the normal baseline window
    # can contain signal and suppress otherwise valid peaks.  Reprocess affected
    # gains with the reference baselines so saturated peaks can fall through to
    # the next usable gain.
    waveforms = (high, mid, low)
    truncated_channels = tuple(
        _baseline_is_truncated(waveform, times, _TOF_REFERENCE_BASELINES[channel_index])
        for channel_index, waveform in enumerate(waveforms)
    )
    if not any(truncated_channels):
        return False

    peak_results = []
    for channel_index, waveform in enumerate(waveforms):
        if truncated_channels[channel_index]:
            peak_results.append(
                _reference_qualifying_peaks(
                    waveform,
                    times,
                    _TOF_REFERENCE_BASELINES[channel_index],
                    _TOF_REFERENCE_SIGMAS[channel_index],
                )
            )
        else:
            peak_results.append(_qualifying_peaks(waveform, times))
    corrected_waveforms: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        peak_results[0][0],
        peak_results[1][0],
        peak_results[2][0],
    )
    peak_sets: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        _collapse_saturated_peaks(peak_results[0][2], high, peak_results[0][0]),
        _collapse_saturated_peaks(peak_results[1][2], mid, peak_results[1][0]),
        _collapse_saturated_peaks(peak_results[2][2], low, peak_results[2][0]),
    )

    for channel_index, (peak_result, peaks) in enumerate(
        zip(peak_results, peak_sets, strict=True)
    ):
        _, sigma, _ = peak_result
        if not np.isfinite(sigma):
            continue
        widths = [
            _gain_aware_width(
                channel_index,
                peak_index,
                waveforms,
                times,
                corrected_waveforms,
                peak_sets,
            )
            for peak_index in peaks
        ]
        if sum(width >= _MIN_PEAK_WIDTH_US for width in widths) >= _MIN_PEAK_COUNT:
            return True

    return False


def _as_1d_array(values: np.ndarray) -> np.ndarray:
    """Convert an event waveform or time coordinate to one dimension.

    Parameters
    ----------
    values : numpy.ndarray
        Input waveform or time coordinate.

    Returns
    -------
    numpy.ndarray
        One-dimensional floating-point array.
    """
    return np.asarray(values, dtype=float).reshape(-1)


def _baseline_corrected(
    values: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, float]:
    """Subtract the baseline and estimate its robust standard deviation.

    Parameters
    ----------
    values : numpy.ndarray
        Waveform samples.
    times : numpy.ndarray
        Sample times in microseconds.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Baseline-corrected samples and estimated noise standard deviation.
    """
    finite = np.isfinite(values) & np.isfinite(times)
    if not np.any(finite):
        return np.full(values.shape, np.nan), np.nan
    first_time = float(times[finite][0])
    baseline_mask = finite & (times < first_time + _BASELINE_WINDOW_US)
    samples = values[baseline_mask]
    if samples.size == 0:
        samples = values[finite]
    baseline = float(np.nanmedian(samples))
    deviations = samples - baseline
    sigma = 1.4826 * float(np.nanmedian(np.abs(deviations)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(samples))
    return values - baseline, sigma


def _find_qualifying_peak_indices(
    corrected: np.ndarray, times: np.ndarray, sigma: float
) -> np.ndarray:
    """Find qualifying peaks in a baseline-corrected waveform.

    Parameters
    ----------
    corrected : numpy.ndarray
        Baseline-corrected waveform samples.
    times : numpy.ndarray
        Sample times in microseconds.
    sigma : float
        Noise standard deviation used to set the peak thresholds.

    Returns
    -------
    numpy.ndarray
        Indices of peaks that satisfy the height, prominence, and separation
        requirements.
    """
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.array([], dtype=int)
    finite = np.isfinite(corrected) & np.isfinite(times)
    if not np.any(finite):
        return np.array([], dtype=int)
    dt_us = _sample_spacing_us(times)
    distance = max(1, round(_MIN_PEAK_DISTANCE_US / dt_us)) if dt_us > 0 else 1
    search = np.where(finite, corrected, -np.inf)
    peaks, _ = find_peaks(
        search,
        height=_PEAK_THRESHOLD_SIGMA * sigma,
        prominence=_PEAK_PROMINENCE_SIGMA * sigma,
        distance=distance,
    )
    return peaks


def _qualifying_peaks(
    values: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return baseline-corrected values and qualifying peaks.

    Parameters
    ----------
    values : numpy.ndarray
        Waveform samples.
    times : numpy.ndarray
        Sample times in microseconds.

    Returns
    -------
    tuple[numpy.ndarray, float, numpy.ndarray]
        Baseline-corrected waveform, noise standard deviation, and peak indices.
    """
    corrected, sigma = _baseline_corrected(values, times)
    peaks = _find_qualifying_peak_indices(corrected, times, sigma)
    return corrected, sigma, peaks


def _baseline_is_truncated(
    values: np.ndarray, times: np.ndarray, reference_baseline: float
) -> bool:
    """Return whether the initial baseline window is signal-contaminated.

    Parameters
    ----------
    values : numpy.ndarray
        Waveform samples.
    times : numpy.ndarray
        Sample times in microseconds.
    reference_baseline : float
        Expected TOF baseline in DN.

    Returns
    -------
    bool
        Whether the initial baseline differs substantially from the reference.
    """
    finite = np.isfinite(values) & np.isfinite(times)
    if not np.any(finite):
        return False
    first_time = float(times[finite][0])
    samples = values[finite & (times < first_time + _BASELINE_WINDOW_US)]
    if samples.size == 0:
        return False
    median = float(np.nanmedian(samples))
    return bool(abs(median - reference_baseline) > _TRUNCATED_BASELINE_OFFSET_DN)


def _reference_qualifying_peaks(
    values: np.ndarray,
    times: np.ndarray,
    reference_baseline: float,
    reference_sigma: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Find peaks using a reference baseline when the record starts in signal.

    Parameters
    ----------
    values : numpy.ndarray
        Waveform samples.
    times : numpy.ndarray
        Sample times in microseconds.
    reference_baseline : float
        Expected TOF baseline in DN.
    reference_sigma : float
        Expected noise standard deviation in DN.

    Returns
    -------
    tuple[numpy.ndarray, float, numpy.ndarray]
        Reference-baseline-corrected waveform, noise standard deviation, and peak
        indices.
    """
    corrected = values - reference_baseline
    finite = np.isfinite(corrected) & np.isfinite(times)
    sigma = (
        reference_sigma
        if np.any(finite) and np.isfinite(reference_sigma) and reference_sigma > 0.0
        else np.nan
    )
    peaks = _find_qualifying_peak_indices(corrected, times, sigma)
    return corrected, sigma, peaks


def _sample_spacing_us(times: np.ndarray) -> float:
    """Return the median finite sample spacing in microseconds.

    Parameters
    ----------
    times : numpy.ndarray
        Sample times in microseconds.

    Returns
    -------
    float
        Median sample spacing, or NaN when fewer than two samples are finite.
    """
    finite_times = times[np.isfinite(times)]
    if finite_times.size < 2:
        return np.nan
    return float(np.nanmedian(np.abs(np.diff(finite_times))))


def _collapse_saturated_peaks(
    peaks: np.ndarray, values: np.ndarray, corrected: np.ndarray
) -> np.ndarray:
    """Keep one candidate peak for each contiguous saturated region.

    Parameters
    ----------
    peaks : numpy.ndarray
        Candidate peak indices.
    values : numpy.ndarray
        Raw waveform samples.
    corrected : numpy.ndarray
        Baseline-corrected waveform samples.

    Returns
    -------
    numpy.ndarray
        Peak indices with saturated regions collapsed to one candidate each.
    """
    if peaks.size < 2:
        return peaks
    saturated = values >= _SATURATION_FRACTION * _TOF_MAX_DN
    retained = []
    used_regions = set()
    for peak_value in peaks:
        peak = int(peak_value)
        if not saturated[peak]:
            retained.append(peak)
            continue
        left = peak
        while left > 0 and saturated[left - 1]:
            left -= 1
        right = peak
        while right < values.size - 1 and saturated[right + 1]:
            right += 1
        region = (left, right)
        if region in used_regions:
            continue
        used_regions.add(region)
        region_peaks = [candidate for candidate in peaks if left <= candidate <= right]
        retained.append(max(region_peaks, key=lambda candidate: corrected[candidate]))
    return np.asarray(retained, dtype=int)


def _saturation_aware_width(
    peak_index: int,
    high: np.ndarray,
    mid: np.ndarray,
    low: np.ndarray,
    times: np.ndarray,
    high_corrected: np.ndarray,
) -> float:
    """Measure a saturated peak width using the first usable gain.

    Parameters
    ----------
    peak_index : int
        High-gain peak index.
    high, mid, low : numpy.ndarray
        Raw TOF waveforms for the three gains.
    times : numpy.ndarray
        Sample times in microseconds.
    high_corrected : numpy.ndarray
        Baseline-corrected high-gain waveform.

    Returns
    -------
    float
        Full width at half maximum in microseconds, or NaN if unavailable.
    """
    if (
        high.size != mid.size
        or high.size != low.size
        or high.size != times.size
        or high_corrected.size != high.size
    ):
        return np.nan

    waveforms = (high, mid, low)
    peak_results = (
        (high_corrected, np.nan, np.array([], dtype=int)),
        _qualifying_peaks(mid, times),
        _qualifying_peaks(low, times),
    )
    corrected_waveforms: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        peak_results[0][0],
        peak_results[1][0],
        peak_results[2][0],
    )
    peak_sets: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        peak_results[0][2],
        peak_results[1][2],
        peak_results[2][2],
    )
    return _gain_aware_width(
        0,
        peak_index,
        waveforms,
        times,
        corrected_waveforms,
        peak_sets,
    )


def _gain_aware_width(
    channel_index: int,
    peak_index: int,
    waveforms: tuple[np.ndarray, np.ndarray, np.ndarray],
    times: np.ndarray,
    corrected_waveforms: tuple[np.ndarray, np.ndarray, np.ndarray],
    peak_sets: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """Measure a peak width with saturation-aware lower-gain fallthrough.

    Parameters
    ----------
    channel_index : int
        Index of the candidate gain channel: High, Mid, or Low.
    peak_index : int
        Index of the candidate peak in the source waveform.
    waveforms : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Raw High-, Mid-, and Low-gain TOF waveforms.
    times : numpy.ndarray
        High-rate waveform times in microseconds.
    corrected_waveforms : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Baseline-corrected High-, Mid-, and Low-gain waveforms.
    peak_sets : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Qualifying peak indices for the High-, Mid-, and Low-gain waveforms.

    Returns
    -------
    float
        Direct or lower-gain-substituted FWHM in microseconds, or NaN when
        no valid width is available.
    """
    if (
        channel_index < 0
        or channel_index >= len(waveforms)
        or peak_index < 0
        or peak_index >= waveforms[channel_index].size
        or any(
            waveform.size != times.size
            for waveform in (*waveforms, *corrected_waveforms)
        )
    ):
        return np.nan

    peak_time = float(times[peak_index])
    if not np.isfinite(peak_time):
        return np.nan
    if not _is_saturated(float(waveforms[channel_index][peak_index])):
        return _fwhm(corrected_waveforms[channel_index], times, peak_index)

    for fallback_index in range(channel_index + 1, len(waveforms)):
        peaks = peak_sets[fallback_index]
        if peaks.size == 0:
            continue
        distances = np.abs(times[peaks] - peak_time)
        nearest = int(np.argmin(distances))
        if (
            not np.isfinite(distances[nearest])
            or distances[nearest] > _SATURATION_MATCH_WINDOW_US
        ):
            continue
        fallback_peak = int(peaks[nearest])
        if _is_saturated(float(waveforms[fallback_index][fallback_peak])):
            continue
        width = _fwhm(corrected_waveforms[fallback_index], times, fallback_peak)
        if np.isfinite(width):
            return width
    return np.nan


def _is_saturated(value: float) -> bool:
    """Return whether a TOF sample exceeds the 95 percent limit.

    Parameters
    ----------
    value : float
        Raw TOF sample in DN.

    Returns
    -------
    bool
        Whether the sample is saturated.
    """
    return bool(value >= _SATURATION_FRACTION * _TOF_MAX_DN)


def _fwhm(corrected: np.ndarray, times: np.ndarray, peak_index: int) -> float:
    """Measure a peak's full width at half maximum.

    Parameters
    ----------
    corrected : numpy.ndarray
        Baseline-corrected waveform.
    times : numpy.ndarray
        Sample times in microseconds.
    peak_index : int
        Index of the peak maximum.

    Returns
    -------
    float
        Full width at half maximum in microseconds, or NaN if unavailable.
    """
    if (
        corrected.ndim != 1
        or times.ndim != 1
        or corrected.size != times.size
        or peak_index < 0
        or peak_index >= corrected.size
    ):
        return np.nan

    peak_height = float(corrected[peak_index])
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        return np.nan
    half_height = peak_height / 2.0
    left = peak_index
    while left > 0 and np.isfinite(corrected[left]) and corrected[left] >= half_height:
        left -= 1
    right = peak_index
    while (
        right < corrected.size - 1
        and np.isfinite(corrected[right])
        and corrected[right] >= half_height
    ):
        right += 1
    left_bracketed = (
        left < corrected.size - 1
        and np.isfinite(corrected[left])
        and corrected[left] < half_height
        and np.isfinite(corrected[left + 1])
        and corrected[left + 1] >= half_height
    )
    right_bracketed = (
        right < corrected.size - 1
        and np.isfinite(corrected[right - 1])
        and corrected[right - 1] >= half_height
        and np.isfinite(corrected[right])
        and corrected[right] < half_height
    )
    if not left_bracketed or not right_bracketed:
        return np.nan
    left_time = _crossing_time(corrected, times, left, left + 1, half_height)
    right_time = _crossing_time(corrected, times, right - 1, right, half_height)
    if not np.isfinite(left_time) or not np.isfinite(right_time):
        return np.nan
    return abs(right_time - left_time)


def _crossing_time(
    values: np.ndarray, times: np.ndarray, low: int, high: int, target: float
) -> float:
    """Linearly interpolate a waveform crossing time.

    Parameters
    ----------
    values : numpy.ndarray
        Waveform values.
    times : numpy.ndarray
        Sample times in microseconds.
    low : int
        Index on the lower side of the crossing.
    high : int
        Index on the upper side of the crossing.
    target : float
        Crossing value.

    Returns
    -------
    float
        Interpolated crossing time, or NaN for invalid samples.
    """
    y0, y1 = values[low], values[high]
    t0, t1 = times[low], times[high]
    if not all(np.isfinite(value) for value in (y0, y1, t0, t1)):
        return np.nan
    if y1 == y0:
        return float(t0)
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))
