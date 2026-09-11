"""Test IDEX event classification and Dust Hit flags."""

import numpy as np
import pytest

from imap_processing.idex.idex_event_flags import (
    ALL_FLAG_NAMES,
    EVENT_FLAG_NAMES,
    SATURATION_FLAG_NAMES,
    _fwhm,
    _qualifying_peaks,
    _reference_qualifying_peaks,
    _saturation_aware_width,
    classify_event_flags,
    classify_saturation_flags,
)
from imap_processing.idex.idex_utils import get_idex_attrs


def _waveforms(saturated: bool = False) -> tuple[np.ndarray, ...]:
    """Create two deterministic, two-peak TOF waveform events."""
    times = np.arange(2048, dtype=float) / 260.0
    baseline = 100.0 + 0.5 * np.sin(np.arange(times.size, dtype=float) / 3.0)
    widths = 0.030 / 2.355
    peaks = sum(
        20.0 * np.exp(-0.5 * ((times - center) / widths) ** 2) for center in (5.0, 5.08)
    )
    high = baseline + peaks
    if saturated:
        high = np.minimum(baseline + 60.0 * peaks, 1023.0)
    medium = baseline + peaks
    low = baseline + peaks
    return high, medium, low, times


def _telemetry(
    *, trigger_id: int = 0, hg_mode: int = 0, hg_threshold: int = 0
) -> dict[str, int]:
    """Return the raw trigger fields used by the classifier."""
    return {
        "idx__txhdrtrigid": trigger_id,
        "idx__txhdrhgtrigmode": hg_mode,
        "idx__txhdrmgtrigmode": 0,
        "idx__txhdrlgtrigmode": 0,
        "idx__txhdrhgtrigctrl1": hg_threshold << 22,
    }


def test_core_event_flags_are_mutually_exclusive() -> None:
    """Exactly one of Science, Noise Capture, and Pulser is set."""
    waveforms = _waveforms()
    cases = (
        _telemetry(),
        _telemetry(trigger_id=1, hg_mode=1, hg_threshold=1000),
        _telemetry(trigger_id=1 | 4, hg_mode=1),
    )

    for telemetry in cases:
        flags = classify_event_flags(telemetry, *waveforms)
        assert sum(flags[name] for name in EVENT_FLAG_NAMES[:3]) == 1
        assert set(flags.values()) <= {0, 1}


def test_event_type_classification() -> None:
    """Classify noise, pulser, and science events from raw trigger fields."""
    waveforms = _waveforms()

    noise = classify_event_flags(_telemetry(), *waveforms)
    assert noise["noise_capture_flag"] == 1

    pulser = classify_event_flags(
        _telemetry(trigger_id=1, hg_mode=1, hg_threshold=1000), *waveforms
    )
    assert pulser["pulser_flag"] == 1
    assert pulser["dust_hit_flag"] == 0

    science = classify_event_flags(_telemetry(trigger_id=1 | 4, hg_mode=1), *waveforms)
    assert science["science_event_flag"] == 1


def test_saturation_flags_use_channel_bit_depth_and_95_percent_limit() -> None:
    """TOF uses 10-bit DN while low-rate channels use 12-bit DN."""
    tof = np.array([0.0, 1023.0 * 0.95])
    low_rate = np.array([0.0, 4095.0 * 0.95])
    flags = classify_saturation_flags(tof, tof, tof, low_rate, low_rate, low_rate)

    assert set(flags) == set(SATURATION_FLAG_NAMES)
    assert all(value == 1 for value in flags.values())

    flags = classify_saturation_flags(
        np.array([1023.0 * 0.95 - 1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([4095.0 * 0.95 - 1.0]),
        None,
        None,
    )
    assert flags["tof_high_saturation_flag"] == 0
    assert flags["target_high_saturation_flag"] == 0
    assert flags["target_low_saturation_flag"] == 0
    assert flags["ion_grid_saturation_flag"] == 0


def test_event_flags_have_istp_integer_flag_attributes() -> None:
    """All event and saturation flags use explicit ISTP-compatible metadata."""
    for level in ("l1a", "l1b"):
        attributes = get_idex_attrs(level)
        for flag_name in ALL_FLAG_NAMES:
            flag_attrs = attributes.get_variable_attributes(flag_name)
            assert flag_attrs["FILLVAL"] == 255
            assert flag_attrs["FORMAT"] == "I1"
            assert flag_attrs["UNITS"] == " "
            assert flag_attrs["VALIDMIN"] == 0
            assert flag_attrs["VALIDMAX"] == 1
            if level == "l1a":
                assert flag_attrs["VAR_TYPE"] == "support_data"
                assert flag_attrs["DISPLAY_TYPE"] == "no_plot"


def test_dust_hit_requires_two_seven_sigma_peaks_and_is_saturation_aware() -> None:
    """Two qualifying peaks set Dust Hit, including saturated High fallback."""
    saturated_waveforms = _waveforms(saturated=True)
    flags = classify_event_flags(
        _telemetry(trigger_id=1 | 4, hg_mode=1), *saturated_waveforms
    )
    assert flags["science_event_flag"] == 1
    assert flags["dust_hit_flag"] == 1


def test_dust_hit_scans_lower_gain_for_two_peaks() -> None:
    """Two peaks visible only in Mid still set Dust Hit when High saturates."""
    times = np.arange(2048, dtype=float) / 260.0
    baseline = 100.0 + 0.5 * np.sin(np.arange(times.size, dtype=float) / 3.0)
    narrow = 0.020 / 2.355
    mid = baseline + sum(
        20.0 * np.exp(-0.5 * ((times - center) / narrow) ** 2) for center in (5.0, 5.08)
    )
    high = np.minimum(baseline + 100.0 * mid, 1023.0)
    low = baseline.copy()

    flags = classify_event_flags(
        _telemetry(trigger_id=1 | 4, hg_mode=1), high, mid, low, times
    )

    assert flags["dust_hit_flag"] == 1


@pytest.mark.parametrize(
    ("low_width", "expected_dust_hit"),
    [(0.010, False), (0.030, True)],
)
def test_saturated_mid_peaks_fall_through_to_low_gain(
    low_width: float, expected_dust_hit: bool
) -> None:
    """Saturated Mid peaks use Low width before passing the FWHM threshold."""
    times = np.arange(2048, dtype=float) / 260.0
    baseline = 511.0 + 0.5 * np.sin(np.arange(times.size, dtype=float) / 3.0)
    narrow_peaks = sum(
        200.0 * np.exp(-0.5 * ((times - center) / (0.010 / 2.355)) ** 2)
        for center in (5.0, 5.08)
    )
    mid = np.minimum(baseline + 10.0 * narrow_peaks, 1023.0)
    low_peaks = sum(
        30.0 * np.exp(-0.5 * ((times - center) / (low_width / 2.355)) ** 2)
        for center in (5.0, 5.08)
    )
    low = baseline + low_peaks
    high = baseline.copy()

    flags = classify_event_flags(
        _telemetry(trigger_id=1 | 4, hg_mode=1), high, mid, low, times
    )

    assert flags["dust_hit_flag"] == int(expected_dust_hit)


def test_truncated_saturated_mid_peaks_fall_through_to_low_gain() -> None:
    """The truncated-baseline path also uses Low width for saturated Mid peaks."""
    times = np.arange(2048, dtype=float) / 260.0
    baseline = 511.0 + 0.5 * np.sin(np.arange(times.size, dtype=float) / 3.0)
    narrow_peaks = sum(
        200.0 * np.exp(-0.5 * ((times - center) / (0.010 / 2.355)) ** 2)
        for center in (5.0, 5.08)
    )
    mid = np.minimum(baseline + 10.0 * narrow_peaks, 1023.0)
    low = baseline + sum(
        30.0 * np.exp(-0.5 * ((times - center) / (0.010 / 2.355)) ** 2)
        for center in (5.0, 5.08)
    )
    high = baseline + 80.0

    flags = classify_event_flags(
        _telemetry(trigger_id=1 | 4, hg_mode=1), high, mid, low, times
    )

    assert flags["dust_hit_flag"] == 0


def test_dust_hit_accepts_one_broad_peak() -> None:
    """A single 50 ns peak is accepted as a broad dust-like event."""
    times = np.arange(2048, dtype=float) / 260.0
    baseline = 100.0 + 0.5 * np.sin(np.arange(times.size, dtype=float) / 3.0)
    broad = baseline + 20.0 * np.exp(-0.5 * ((times - 5.0) / (0.050 / 2.355)) ** 2)

    flags = classify_event_flags(
        _telemetry(trigger_id=1 | 4, hg_mode=1), broad, broad, broad, times
    )

    assert flags["dust_hit_flag"] == 1


def test_dust_hit_uses_reference_baseline_for_truncated_waveform() -> None:
    """Two peaks are recovered when the initial TOF baseline is contaminated."""
    times = np.arange(2048, dtype=float) / 260.0
    baseline = np.full(times.size, 511.0)
    width = 0.030 / 2.355
    waveform = baseline + 80.0
    waveform += sum(
        200.0 * np.exp(-0.5 * ((times - center) / width) ** 2) for center in (5.0, 5.08)
    )
    _, _, normal_peaks = _qualifying_peaks(waveform, times)
    _, _, fallback_peaks = _reference_qualifying_peaks(waveform, times, 511.0, 2.9652)

    assert normal_peaks.size < 2
    assert fallback_peaks.size >= 2

    flags = classify_event_flags(
        _telemetry(trigger_id=1 | 4, hg_mode=1),
        waveform,
        waveform - 2.0,
        waveform - 1.0,
        times,
    )
    assert flags["dust_hit_flag"] == 1


def test_noisy_baseline_does_not_use_reference_noise_sigma() -> None:
    """A globally noisy baseline is not reprocessed as a truncated record."""
    times = np.arange(2048, dtype=float) / 260.0
    noisy_baseline = 511.0 + 25.0 * np.sin(np.arange(times.size, dtype=float) / 3.0)

    flags = classify_event_flags(
        _telemetry(trigger_id=1 | 4, hg_mode=1),
        noisy_baseline,
        noisy_baseline,
        noisy_baseline,
        times,
    )

    assert flags["dust_hit_flag"] == 0


def test_peak_detection_rejects_low_prominence_secondary_maxima() -> None:
    """A small shoulder above 7 sigma is not counted as a separate peak."""
    times = np.arange(256, dtype=float) / 260.0
    waveform = 0.5 * np.sin(np.arange(times.size, dtype=float) / 3.0)
    waveform[90:130] += 6.0
    waveform[100] += 20.0
    waveform[104:] += 7.5
    waveform[108] += 0.5

    _, _, peaks = _qualifying_peaks(waveform, times)

    assert peaks.tolist() == [100]


def test_dust_hit_is_not_set_for_non_science_events() -> None:
    """Dust-shaped waveforms cannot turn a non-science event into Dust Hit."""
    flags = classify_event_flags(
        _telemetry(trigger_id=1, hg_mode=1, hg_threshold=1000), *_waveforms()
    )
    assert flags["pulser_flag"] == 1
    assert flags["dust_hit_flag"] == 0


def test_saturation_aware_width_falls_through_invalid_mid_gain() -> None:
    """A non-finite Mid sample falls through to a usable Low waveform."""
    times = np.arange(2048, dtype=float) / 260.0
    baseline = 100.0 + 0.5 * np.sin(np.arange(times.size, dtype=float) / 3.0)
    low = baseline + 20.0 * np.exp(-0.5 * ((times - 5.0) / (0.030 / 2.355)) ** 2)
    high = low.copy()
    peak_index = int(np.argmax(low))
    high[peak_index] = 1023.0
    mid = np.full_like(low, np.nan)

    width = _saturation_aware_width(peak_index, high, mid, low, times, high - high[0])

    assert width == pytest.approx(0.030, rel=0.1)


def test_fwhm_rejects_truncated_boundary_peaks() -> None:
    """A missing half-height crossing at either edge is not measurable."""
    times = np.arange(4, dtype=float)
    assert np.isnan(_fwhm(np.array([2.0, 2.0, 1.0, 0.0]), times, 1))
    assert np.isnan(_fwhm(np.array([0.0, 1.0, 2.0, 2.0]), times, 2))


def test_saturation_aware_width_rejects_invalid_peak_inputs() -> None:
    """Invalid waveform lengths or times produce no measurable peak."""
    values = np.ones(4)
    assert np.isnan(
        _saturation_aware_width(
            1, values, values, values[:-1], np.arange(4, dtype=float), values
        )
    )
    assert np.isnan(
        _saturation_aware_width(
            1,
            values,
            values,
            values,
            np.full(4, np.nan),
            values,
        )
    )
