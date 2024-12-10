"""Tests coverage for imap_processing/spice/time.py"""

import numpy as np
import pytest
import spiceypy

from imap_processing.spice import IMAP_SC_ID
from imap_processing.spice.time import (
    TICK_DURATION,
    _sct2e_wrapper,
    et_to_utc,
    j2000ns_to_j2000s,
    met_to_datetime64,
    met_to_j2000ns,
    met_to_sclkticks,
    met_to_utc,
    str_to_et,
)


@pytest.mark.parametrize("met", [1, np.arange(10)])
def test_met_to_sclkticks(met):
    """Test coverage for met_to_sclkticks."""
    # Tick duration is 20us as specified in imap_sclk_0000.tsc
    expected = met * 1 / 20e-6
    ticks = met_to_sclkticks(met)
    np.testing.assert_array_equal(ticks, expected)


def test_met_to_j2000ns(furnish_time_kernels):
    """Test coverage for met_to_j2000ns function."""
    utc = "2026-01-01T00:00:00.125"
    et = spiceypy.str2et(utc)
    sclk_str = spiceypy.sce2s(IMAP_SC_ID, et)
    seconds, ticks = sclk_str.split("/")[1].split(":")
    # There is some floating point error calculating tick duration from 1 clock
    # tick so average over many clock ticks for better accuracy
    spice_tick_duration = (
        spiceypy.sct2e(IMAP_SC_ID, 1e12) - spiceypy.sct2e(IMAP_SC_ID, 0)
    ) / 1e12
    met = float(seconds) + float(ticks) * spice_tick_duration
    j2000ns = met_to_j2000ns(met)
    assert j2000ns.dtype == np.int64
    np.testing.assert_array_equal(j2000ns, np.array(et * 1e9))


def test_j2000ns_to_j2000s(furnish_time_kernels):
    """Test coverage for j2000ns_to_j2000s function."""
    # Use spice to come up with reasonable J2000 values
    utc = "2025-09-23T00:00:00.000"
    # Test single value input
    et = spiceypy.str2et(utc)
    epoch = int(et * 1e9)
    j2000s = j2000ns_to_j2000s(epoch)
    assert j2000s == et
    # Test array input
    epoch = (np.arange(et, et + 10000, 100) * 1e9).astype(np.int64)
    j2000s = j2000ns_to_j2000s(epoch)
    np.testing.assert_array_equal(
        j2000s, np.arange(et, et + 10000, 100, dtype=np.float64)
    )


@pytest.mark.parametrize(
    "expected_utc, precision",
    [
        ("2024-01-01T00:00:00.000", 3),
        (
            [
                "2024-01-01T00:00:00.000555",
                "2025-09-23T00:00:00.000111",
                "2040-11-14T10:23:48.156980",
            ],
            6,
        ),
    ],
)
def test_met_to_utc(furnish_time_kernels, expected_utc, precision):
    """Test coverage for met_to_utc function."""
    if isinstance(expected_utc, list):
        et_arr = spiceypy.str2et(expected_utc)
        sclk_ticks = np.array([spiceypy.sce2c(IMAP_SC_ID, et) for et in et_arr])
    else:
        et = spiceypy.str2et(expected_utc)
        sclk_ticks = spiceypy.sce2c(IMAP_SC_ID, et)
    met = sclk_ticks * TICK_DURATION
    utc = met_to_utc(met, precision=precision)
    np.testing.assert_array_equal(utc, expected_utc)


@pytest.mark.parametrize(
    "utc",
    [
        "2024-01-01T00:00:00.000",
        [
            "2024-01-01T00:00:00.000",
            "2025-09-23T00:00:00.000",
            "2040-11-14T10:23:48.15698",
        ],
    ],
)
def test_met_to_datetime64(furnish_time_kernels, utc):
    """Test coverage for met_to_datetime64 function."""
    if isinstance(utc, list):
        expected_dt64 = np.array([np.datetime64(utc_str) for utc_str in utc])
        et_arr = spiceypy.str2et(utc)
        sclk_ticks = np.array([spiceypy.sce2c(IMAP_SC_ID, et) for et in et_arr])
    else:
        expected_dt64 = np.asarray(np.datetime64(utc))
        et = spiceypy.str2et(utc)
        sclk_ticks = spiceypy.sce2c(IMAP_SC_ID, et)
    met = sclk_ticks * TICK_DURATION
    dt64 = met_to_datetime64(met)
    np.testing.assert_array_equal(
        dt64.astype("datetime64[us]"), expected_dt64.astype("datetime64[us]")
    )


@pytest.mark.parametrize("sclk_ticks", [0.0, np.arange(10)])
def test_sct2e_wrapper(sclk_ticks):
    """Test for `_sct2e_wrapper` function."""
    et = _sct2e_wrapper(sclk_ticks)
    if isinstance(sclk_ticks, float):
        assert isinstance(et, float)
    else:
        assert len(et) == len(sclk_ticks)


def test_str_to_et(furnish_time_kernels):
    """Test coverage for string to et conversion function."""
    utc = "2017-07-14T19:46:00"
    # Test single value input
    expected_et = 553333629.1837274
    actual_et = str_to_et(utc)
    assert expected_et == actual_et

    # Test list input
    list_of_utc = [
        "2017-08-14T19:46:00.000",
        "2017-09-14T19:46:00.000",
        "2017-10-14T19:46:00.000",
    ]

    expected_et_array = np.array(
        (556012029.1829445, 558690429.1824446, 561282429.1823651)
    )
    actual_et_array = str_to_et(list_of_utc)
    assert np.array_equal(expected_et_array, actual_et_array)

    # Test array input
    array_of_utc = np.array(
        [
            "2017-08-14T19:46:00.000",
            "2017-09-14T19:46:00.000",
            "2017-10-14T19:46:00.000",
        ]
    )

    actual_et_array = str_to_et(array_of_utc)
    assert np.array_equal(expected_et_array, actual_et_array)


def test_et_to_utc(furnish_time_kernels):
    """Test coverage for et to utc conversion function."""
    et = 553333629.1837274
    # Test single value input
    expected_utc = "2017-07-14T19:46:00.000"
    actual_utc = et_to_utc(et)
    assert expected_utc == actual_utc

    # Test array input
    array_of_et = np.array((556012029.1829445, 558690429.1824446, 561282429.1823651))
    expected_utc_array = np.array(
        (
            "2017-08-14T19:46:00.000",
            "2017-09-14T19:46:00.000",
            "2017-10-14T19:46:00.000",
        )
    )
    actual_utc_array = et_to_utc(array_of_et)
    assert np.array_equal(expected_utc_array, actual_utc_array)
