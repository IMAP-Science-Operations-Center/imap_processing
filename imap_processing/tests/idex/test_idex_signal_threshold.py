"""Test the signal threshold for Target and Ion Grid waveform fits."""

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing.idex.idex_l2a import (
    estimate_dust_mass,
    idex_l2a,
    log_smooth_powerlaw,
)


@pytest.mark.parametrize("waveform", ["Target_High", "Target_Low", "Ion_Grid"])
@pytest.mark.parametrize("excursion", [0.0, 1.49, 1.5, 1.51, -1.49, -1.5, -1.51])
@pytest.mark.parametrize("pulse_index", [6, -1])
def test_low_rate_fit_requires_sample_above_noise(waveform, excursion, pulse_index):
    """One sample must strictly exceed 1.5 sigma, with channel-specific polarity."""
    time = xr.DataArray(np.arange(-6.0, 6.0, 0.5))
    signal = np.full(time.size, 10.0)
    # The first 3 microseconds have baseline 10 and population sigma 1.
    signal[:6] += [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    signal[pulse_index] += excursion
    target = xr.DataArray(signal, name=waveform)
    parameters = np.array([0.0, 10.0, excursion, 0.371, 37.1])

    with mock.patch(
        "imap_processing.idex.idex_l2a.curve_fit", return_value=(parameters, None)
    ) as optimizer:
        result = estimate_dust_mass(time, target)

    should_fit = abs(excursion) > 1.5 if waveform == "Ion_Grid" else excursion > 1.5
    if should_fit:
        optimizer.assert_called_once()
        np.testing.assert_array_equal(result[0], parameters)
        assert np.all(np.isfinite(result[4]))
    else:
        optimizer.assert_not_called()
        assert result[0].shape == (5,)
        assert result[4].shape == signal.shape
        for value in result:
            assert np.all(np.isnan(value))


@pytest.mark.parametrize("waveform", ["Target_High", "Target_Low", "Ion_Grid"])
@pytest.mark.parametrize("excursion", [0.0, 1.0, -1.0])
def test_low_rate_fit_zero_noise(waveform, excursion):
    """A flat baseline requires a nonzero excursion without dividing by zero."""
    time = xr.DataArray(np.arange(-6.0, 6.0, 0.5))
    signal = np.full(time.size, -2.0)
    signal[-1] += excursion
    with mock.patch(
        "imap_processing.idex.idex_l2a.curve_fit",
        return_value=(np.array([0.0, -2.0, excursion, 0.371, 37.1]), None),
    ) as optimizer:
        result = estimate_dust_mass(time, xr.DataArray(signal), waveform_name=waveform)

    should_fit = excursion != 0.0 if waveform == "Ion_Grid" else excursion > 0.0
    assert optimizer.call_count == int(should_fit)
    if not should_fit:
        for value in result:
            assert np.all(np.isnan(value))


@pytest.mark.parametrize("waveform", ["Target_High", "Target_Low", "Ion_Grid"])
def test_low_rate_fit_does_not_count_baseline_excursions(waveform):
    """An excursion confined to the noise window is not evidence of a pulse."""
    time = xr.DataArray(np.arange(-6.0, 6.0, 0.5))
    signal = np.full(time.size, 4.0 / 6.0)
    signal[:6] = [0.0, 0.0, 0.0, 0.0, 0.0, 4.0]
    with mock.patch("imap_processing.idex.idex_l2a.curve_fit") as optimizer:
        result = estimate_dust_mass(time, xr.DataArray(signal), waveform_name=waveform)

    optimizer.assert_not_called()
    for value in result:
        assert np.all(np.isnan(value))


@pytest.mark.parametrize("missing", ["signal", "time", "baseline"])
def test_low_rate_fit_requires_finite_noise_samples(missing):
    """Missing baseline information cannot establish a significant signal."""
    time = np.arange(-6.0, 6.0, 0.5)
    signal = np.zeros(time.size)
    signal[-1] = 10.0
    if missing == "signal":
        signal[:] = np.nan
    elif missing == "time":
        time[:] = np.nan
    else:
        signal[:6] = np.nan

    with mock.patch("imap_processing.idex.idex_l2a.curve_fit") as optimizer:
        result = estimate_dust_mass(
            xr.DataArray(time), xr.DataArray(signal), waveform_name="Target_High"
        )

    optimizer.assert_not_called()
    for value in result:
        assert np.all(np.isnan(value))


@pytest.mark.parametrize("waveform", ["Target_High", "Target_Low", "Ion_Grid"])
def test_low_rate_signal_gate_is_per_event(waveform):
    """Vectorized fits skip weak events and keep an independent strong event."""
    time = xr.DataArray(np.arange(-6.0, 6.0, 0.5), dims="sample")
    signal = np.zeros((2, time.size))
    signal[:, :6] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    signal[0, -1] = 1.5
    signal[1, -1] = -2.0 if waveform == "Ion_Grid" else 2.0
    with mock.patch(
        "imap_processing.idex.idex_l2a.curve_fit",
        return_value=(np.array([0.0, 0.0, 2.0, 0.371, 37.1]), None),
    ) as optimizer:
        results = xr.apply_ufunc(
            estimate_dust_mass,
            time,
            xr.DataArray(signal, dims=("epoch", "sample")),
            input_core_dims=[["sample"], ["sample"]],
            output_core_dims=[["parameter"], [], [], [], ["sample"]],
            output_dtypes=[np.float64] * 5,
            vectorize=True,
            kwargs={"waveform_name": waveform},
        )

    optimizer.assert_called_once()
    for values in results:
        assert np.all(np.isnan(values.isel(epoch=0)))
        assert np.all(np.isfinite(values.isel(epoch=1)))


def test_l2a_signal_gate_preserves_independent_channels():
    """Skipped fits propagate NaNs while other channels retain their products."""
    time = np.tile(np.arange(-6.0, 6.0, 0.5), (2, 1))
    dimensions = ("epoch", "time_low_sample_rate_index")
    high_dimensions = ("epoch", "time_high_sample_rate_index")
    dataset = xr.Dataset(
        {
            "time_low_sample_rate": (dimensions, time),
            "time_high_sample_rate": (high_dimensions, time - 1.0),
            "TOF_High": (high_dimensions, np.tile([-1.0, 1.0], (2, 12))),
            "science_event_flag": ("epoch", [1, 1]),
            "tof_high_saturation_flag": ("epoch", [0, 0]),
        },
        coords={"epoch": [0, 1]},
        attrs={"Logical_source": "imap_idex_l1b_sci-10days"},
    )
    strong_events = {"Target_High": [0], "Target_Low": [1], "Ion_Grid": [0, 1]}
    for waveform, events in strong_events.items():
        signal = np.zeros_like(time)
        signal[:, :6] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        signal[events, -1] = 10.0
        if waveform == "Ion_Grid":
            signal[0, -1] = -10.0
        dataset[waveform] = xr.DataArray(signal, dims=dimensions)
        dataset[f"{waveform.lower()}_saturation_flag"] = ("epoch", [0, 0])

    t_rise_params = np.array([1.27, -0.2, -2.1, -0.37, 5.3, 13.3, 13.3, 0.28])
    rise_time = 10 ** log_smooth_powerlaw(1.0, t_rise_params[0], t_rise_params[1:])

    def fit_waveform(function, time, signal, p0, **kwargs):
        parameters = np.asarray(p0)
        parameters[3] = rise_time
        return parameters, None

    with (
        mock.patch(
            "imap_processing.idex.idex_l2a.load_calibration_files",
            return_value=(
                t_rise_params,
                np.array([0.06, 2.8, 5.9, 4.1, 13.0, 22.7, 8.2, 0.40]),
            ),
        ),
        mock.patch(
            "imap_processing.idex.idex_l2a.time_to_mass",
            return_value=(None, None, np.zeros_like(time)),
        ),
        mock.patch(
            "imap_processing.idex.idex_l2a.analyze_peaks",
            return_value=(np.zeros((2, 3)), np.ones(2), np.zeros(2), np.zeros(2)),
        ),
        mock.patch(
            "imap_processing.idex.idex_l2a.curve_fit", side_effect=fit_waveform
        ) as optimizer,
    ):
        result = idex_l2a(dataset, {})

    assert optimizer.call_count == 4
    for waveform, events in strong_events.items():
        for product in (
            "fit_parameters",
            "impact_charge",
            "chi_squared",
            "reduced_chi_squared",
            "fit_results",
            "velocity_estimate",
            "dust_mass_estimate",
        ):
            values = result[f"{waveform.lower()}_{product}"]
            assert np.all(np.isfinite(values.isel(epoch=events)))
            weak_events = [event for event in range(2) if event not in events]
            assert np.all(np.isnan(values.isel(epoch=weak_events)))
