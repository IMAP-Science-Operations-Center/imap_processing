"""Tests of shared IDEX waveform statistics."""

import numpy as np
import pytest

from imap_processing.idex.idex_utils import calculate_baseline_and_rms


def test_baseline_and_rms_use_first_three_microseconds_per_event():
    # Offsets and sample intervals vary between events. The endpoint is excluded.
    time = np.array([[-8.0, -7.0, -6.0, -5.0], [4.0, 4.5, 6.9, 7.0]])
    signal = np.array([[2.0, 4.0, 6.0, 1000.0], [12.0, 14.0, 16.0, 2000.0]])
    original = signal.copy()

    baseline, rms = calculate_baseline_and_rms(signal, time)

    np.testing.assert_allclose(baseline, [4.0, 14.0])
    np.testing.assert_allclose(rms, np.sqrt([8 / 3, 8 / 3]))
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize(
    "signal,time,expected_baseline,expected_rms",
    [
        ([5.0, 5.0], [0.0, 1.0], 5.0, 0.0),
        ([7.0], [-2.0], 7.0, 0.0),
        ([np.nan, 2.0, np.inf, 4.0], [-1.0, 0.0, 0.5, 1.0], 3.0, 1.0),
        ([1000.0, 2.0, 4.0], [np.nan, 0.0, 1.0], 3.0, 1.0),
        ([1000.0, 2.0, 4.0], [-np.inf, 0.0, 1.0], 3.0, 1.0),
        # Missing initial signal samples must not move the window into the pulse.
        ([np.nan, 2.0, 1000.0], [0.0, 1.0, 3.0], 2.0, 0.0),
    ],
)
def test_baseline_and_rms_short_or_nonfinite_samples(
    signal, time, expected_baseline, expected_rms
):
    baseline, rms = calculate_baseline_and_rms(np.array(signal), np.array(time))

    np.testing.assert_allclose(baseline, expected_baseline)
    np.testing.assert_allclose(rms, expected_rms)


@pytest.mark.parametrize(
    "signal,time",
    [([np.nan, np.inf], [0.0, 1.0]), ([1.0, 2.0], [np.nan, np.inf]), ([], [])],
)
def test_baseline_and_rms_missing_window(signal, time, caplog):
    baseline, rms = calculate_baseline_and_rms(np.array(signal), np.array(time))

    assert np.isnan(baseline)
    assert np.isnan(rms)
    assert "Unable to find baseline noise" in caplog.text


def test_baseline_and_rms_missing_event_does_not_affect_other_events():
    baseline, rms = calculate_baseline_and_rms(
        np.array([[np.nan, np.nan], [1.0, 3.0]]),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
    )

    np.testing.assert_allclose(baseline, [np.nan, 2.0])
    np.testing.assert_allclose(rms, [np.nan, 1.0])
