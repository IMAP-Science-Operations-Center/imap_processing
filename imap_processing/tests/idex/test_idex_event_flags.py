"""Test IDEX event classification and Dust Hit flags."""

import numpy as np

from imap_processing.idex.idex_event_flags import (
    EVENT_FLAG_NAMES,
    classify_event_flags,
)


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


def test_dust_hit_requires_two_seven_sigma_peaks_and_is_saturation_aware() -> None:
    """Two qualifying peaks set Dust Hit, including saturated High fallback."""
    saturated_waveforms = _waveforms(saturated=True)
    flags = classify_event_flags(
        _telemetry(trigger_id=1 | 4, hg_mode=1), *saturated_waveforms
    )
    assert flags["science_event_flag"] == 1
    assert flags["dust_hit_flag"] == 1


def test_dust_hit_is_not_set_for_non_science_events() -> None:
    """Dust-shaped waveforms cannot turn a non-science event into Dust Hit."""
    flags = classify_event_flags(
        _telemetry(trigger_id=1, hg_mode=1, hg_threshold=1000), *_waveforms()
    )
    assert flags["pulser_flag"] == 1
    assert flags["dust_hit_flag"] == 0
