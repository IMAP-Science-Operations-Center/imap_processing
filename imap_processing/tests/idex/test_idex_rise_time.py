"""Tests for the continuous fitted 10-90 percent rise time."""

import numpy as np
import pytest

from imap_processing.idex.idex_l2a import calculate_rise_time_10_90


@pytest.mark.parametrize("amplitude", [8.0, -8.0])
@pytest.mark.parametrize("offset", [0.0, 1e12])
@pytest.mark.parametrize("time_of_impact", [-20.0, 1e15])
@pytest.mark.parametrize("time_scale", [1e-9, 1.0, 1e9])
def test_equal_time_constants_have_analytic_crossings(
    amplitude, offset, time_of_impact, time_scale
):
    # For tau_r = tau_d, f/A = z(1-z), z = exp(-t/tau), and peak/A = 1/4.
    # Select the leading-edge solution z = (1 + sqrt(1-fraction)) / 2.
    t10 = -np.log((1 + np.sqrt(0.9)) / 2)
    t90 = -np.log((1 + np.sqrt(0.1)) / 2)
    params = np.array([time_of_impact, offset, amplitude, time_scale, time_scale])
    result = calculate_rise_time_10_90(params)
    assert result == pytest.approx((t90 - t10) * time_scale, rel=1e-10, abs=0)


def test_slow_decay_approaches_exponential_rise():
    rise = 0.371
    params = np.array([0.0, 3.0, 8.0, rise, rise * 1e12])
    assert calculate_rise_time_10_90(params) == pytest.approx(
        rise * np.log(9), rel=1e-9
    )


@pytest.mark.parametrize("decay_ratio", [1e-6, 0.01, 100.0, 1e6])
def test_rise_time_matches_dense_leading_edge(decay_ratio):
    rise = 0.371
    decay = rise * decay_ratio
    peak_time = rise * np.log1p(decay_ratio)
    times = np.linspace(0, peak_time, 100_001)
    pulse = -np.expm1(-times / rise) * np.exp(-times / decay)
    crossings = np.interp([0.1, 0.9], pulse / pulse[-1], times)
    result = calculate_rise_time_10_90(np.array([2.0, 10.0, -8.0, rise, decay]))
    assert result == pytest.approx(crossings[1] - crossings[0], rel=1e-7)
    assert 0 < result < peak_time


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_parameters_have_no_rise_time(index, value):
    params = np.array([0.0, 1.0, 8.0, 0.371, 37.1])
    params[index] = value
    assert np.isnan(calculate_rise_time_10_90(params))


@pytest.mark.parametrize(
    "index,value", [(2, 0.0), (3, 0.0), (3, -1.0), (4, 0.0), (4, -1.0)]
)
def test_flat_or_invalid_pulse_has_no_rise_time(index, value):
    params = np.array([0.0, 1.0, 8.0, 0.371, 37.1])
    params[index] = value
    assert np.isnan(calculate_rise_time_10_90(params))
