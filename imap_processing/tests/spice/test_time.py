"""Tests coverage for imap_processing/spice/time.py"""

import numpy as np
import pytest
import spiceypy

from imap_processing.spice import IMAP_SC_ID
from imap_processing.spice.time import (
    TICK_DURATION,
    epoch_to_doy,
    epoch_to_fractional_doy,
    et_to_datetime64,
    et_to_met,
    et_to_ttj2000ns,
    et_to_utc,
    met_to_datetime64,
    met_to_sclkticks,
    met_to_ttj2000ns,
    met_to_utc,
    sct_to_et,
    sct_to_ttj2000s,
    str_to_et,
    str_yyyymmdd_to_ttj2000ns,
    ttj2000ns_to_et,
    ttj2000ns_to_met,
)


@pytest.mark.parametrize("met", [1, np.arange(10)])
def test_met_to_sclkticks(met):
    """Test coverage for met_to_sclkticks."""
    # Tick duration is 20us as specified in imap_sclk_0000.tsc
    expected = met * 1 / 20e-6
    ticks = met_to_sclkticks(met)
    np.testing.assert_array_equal(ticks, expected)


def test_met_to_ttj2000ns():
    """Test coverage for met_to_ttj2000ns function."""
    utc = "2026-01-01T00:00:00.125"
    et = spiceypy.str2et(utc)
    spicey_tt = spiceypy.unitim(et, "ET", "TT")
    sclk_str = spiceypy.sce2s(IMAP_SC_ID, et)
    seconds, ticks = sclk_str.split("/")[1].split(":")
    # There is some floating point error calculating tick duration from 1 clock
    # tick so average over many clock ticks for better accuracy
    spice_tick_duration = (
        spiceypy.sct2e(IMAP_SC_ID, 1e12) - spiceypy.sct2e(IMAP_SC_ID, 0)
    ) / 1e12
    met = float(seconds) + float(ticks) * spice_tick_duration
    tt = met_to_ttj2000ns(met)
    assert tt.dtype == np.int64
    np.testing.assert_array_equal(tt, np.array(spicey_tt * 1e9))


def test_ttj2000ns_to_et():
    """Test coverage for ttj2000ns_to_et function."""
    # Use spice to come up with reasonable J2000 values
    utc = "2025-09-23T00:00:00.000"
    # Test single value input
    et = spiceypy.str2et(utc)
    epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)
    j2000s = ttj2000ns_to_et(epoch)
    assert j2000s == et
    # Test for bug when spiceypy tries to iterate over 0-d array returned by
    # np.vectorize for the scalar case
    assert not spiceypy.support_types.is_iterable(et)
    # Test array input
    ets = np.arange(et, et + 10000, 100)
    epoch = np.array([spiceypy.unitim(et, "ET", "TT") * 1e9 for et in ets]).astype(
        np.int64
    )
    j2000s = ttj2000ns_to_et(epoch)
    np.testing.assert_array_equal(j2000s, ets)


def test_et_to_ttj2000ns():
    """Test coverage for ttj2000ns_to_et function."""
    # Use spice to come up with reasonable J2000 values
    utc = "2025-09-23T00:00:00.000"
    # Test single value input
    et = spiceypy.str2et(utc)
    epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)
    j2000ns = et_to_ttj2000ns(et)
    assert j2000ns == epoch

    # Test converting back
    et_roundtrip = ttj2000ns_to_et(j2000ns)
    assert et_roundtrip == et

    # Test for bug when spiceypy tries to iterate over 0-d array returned by
    # np.vectorize for the scalar case
    assert not spiceypy.support_types.is_iterable(et)
    # Test array input
    ets = np.arange(et, et + 10000, 100)
    epoch = np.array([spiceypy.unitim(et, "ET", "TT") * 1e9 for et in ets]).astype(
        np.int64
    )
    j2000ns = et_to_ttj2000ns(ets)
    np.testing.assert_array_equal(j2000ns, epoch)


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
def test_met_to_utc(expected_utc, precision):
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
def test_met_to_datetime64(utc):
    """Test coverage for met_to_datetime64 function."""
    if isinstance(utc, list):
        expected_dt64 = np.array([np.datetime64(utc_str) for utc_str in utc])
        et_arr = spiceypy.str2et(utc)
        sclk_ticks = np.array([spiceypy.sce2c(IMAP_SC_ID, et) for et in et_arr])
    else:
        expected_dt64 = np.datetime64(utc)
        et = spiceypy.str2et(utc)
        sclk_ticks = spiceypy.sce2c(IMAP_SC_ID, et)
    met = sclk_ticks * TICK_DURATION
    dt64 = met_to_datetime64(met)

    if isinstance(utc, list):
        assert isinstance(dt64, np.ndarray)
        assert dt64.dtype.name == "datetime64[ns]"
    else:
        assert isinstance(dt64, np.datetime64)
    np.testing.assert_array_equal(
        dt64.astype("datetime64[us]"), expected_dt64.astype("datetime64[us]")
    )


@pytest.mark.parametrize("sclk_ticks", [0.0, np.arange(10)])
def test_sct_to_et(
    sclk_ticks,
):
    """Test for `sct_to_et` function."""
    et = sct_to_et(sclk_ticks)
    if isinstance(sclk_ticks, float):
        assert isinstance(et, float)
    else:
        assert len(et) == len(sclk_ticks)


@pytest.mark.parametrize("sclk_ticks", [0.0, np.arange(10)])
def test_sct_to_ttj2000s(
    sclk_ticks,
):
    """Test for `sct_to_ttj2000s` function."""
    tt = sct_to_ttj2000s(sclk_ticks)
    if isinstance(sclk_ticks, float):
        assert isinstance(tt, float)
    else:
        assert len(tt) == len(sclk_ticks)


def test_str_to_et():
    """Test coverage for string to et conversion function."""
    utc = "2017-07-14T19:46:00"
    # Test single value input
    expected_et = 553333629.1837274
    actual_et = str_to_et(utc)
    assert expected_et == actual_et
    assert isinstance(actual_et, float)

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


def test_et_to_utc():
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


def test_et_to_datetime():
    et = 553333629.1837274
    # Test single value input
    expected_dt = np.datetime64("2017-07-14T19:46:00.000")

    actual_dt = et_to_datetime64(et)
    assert actual_dt == expected_dt


def test_epoch_to_doy():
    """Tests that epoch_to_doy() produces expected doys."""
    epoch = 756196488384840064
    et = epoch * 1e-9
    # Extract DOY from output date string
    expected_doy = int(et_to_utc(et, "D").split("//")[0].split("-")[1])
    example_epoch = np.full(10, epoch)
    doy = epoch_to_doy(example_epoch)

    # Assert that every calculated DOY is equal to the DOY extracted from the string.
    assert np.all(doy == expected_doy)


def test_et_to_met():
    """Test coverage for et_to_met function."""
    utc = "2026-01-01T00:00:00.125"
    et = spiceypy.str2et(utc)
    sclk_ticks = spiceypy.sce2c(IMAP_SC_ID, et)
    expected_met = sclk_ticks * TICK_DURATION

    # Test single value
    met = et_to_met(et)
    assert isinstance(met, float)
    np.testing.assert_almost_equal(met, expected_met)

    # Test array
    et_array = np.array([et, et + 100, et + 200])
    expected_met_array = np.array(
        [spiceypy.sce2c(IMAP_SC_ID, et_val) * TICK_DURATION for et_val in et_array]
    )
    met_array = et_to_met(et_array)
    np.testing.assert_array_almost_equal(met_array, expected_met_array)


def test_ttj2000ns_to_met():
    """Test coverage for ttj2000ns_to_met function."""
    # Test roundtrip: MET -> TTJ2000ns -> MET
    original_met = 1000.0
    ttj2000ns = met_to_ttj2000ns(original_met)
    roundtrip_met = ttj2000ns_to_met(ttj2000ns)

    # Should get back to original MET (within floating point precision)
    np.testing.assert_almost_equal(roundtrip_met, original_met)

    # Test array input
    met_array = np.array([1000.0, 2000.0, 3000.0])
    ttj2000ns_array = met_to_ttj2000ns(met_array)
    roundtrip_met_array = ttj2000ns_to_met(ttj2000ns_array)
    np.testing.assert_array_almost_equal(roundtrip_met_array, met_array)


class TestEpochToFractionalDoy:
    """Tests for epoch_to_fractional_doy function."""

    def test_january_first_midnight(self):
        """Test that January 1st 00:00:00 returns exactly 1.0."""
        # January 1st at midnight should be DOY 1.0
        utc = "2025-01-01T00:00:00.000"
        et = spiceypy.str2et(utc)
        epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)

        doy = epoch_to_fractional_doy(epoch)

        assert doy == 1.0

    def test_january_first_noon(self):
        """Test that January 1st 12:00:00 returns 1.5."""
        # January 1st at noon should be DOY 1.5
        utc = "2025-01-01T12:00:00.000"
        et = spiceypy.str2et(utc)
        epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)

        doy = epoch_to_fractional_doy(epoch)

        np.testing.assert_almost_equal(doy, 1.5, decimal=6)

    def test_known_mid_year_date(self):
        """Test a known mid-year date (July 1st = DOY 182 in non-leap year)."""
        # July 1st 00:00:00 in 2025 (non-leap year) should be DOY 182.0
        utc = "2025-07-01T00:00:00.000"
        et = spiceypy.str2et(utc)
        epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)

        doy = epoch_to_fractional_doy(epoch)

        # July 1st is day 182 in a non-leap year
        # Jan(31) + Feb(28) + Mar(31) + Apr(30) + May(31) + Jun(30) = 181 days
        # So July 1 is day 182
        assert doy == 182.0

    def test_leap_year_date(self):
        """Test leap year handling (Feb 29th exists in 2024)."""
        # March 1st 00:00:00 in 2024 (leap year) should be DOY 61.0
        utc = "2024-03-01T00:00:00.000"
        et = spiceypy.str2et(utc)
        epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)

        doy = epoch_to_fractional_doy(epoch)

        # In leap year: Jan(31) + Feb(29) = 60 days, so March 1 is day 61
        assert doy == 61.0

    def test_end_of_year(self):
        """Test December 31st returns DOY 365 (non-leap) or 366 (leap)."""
        # December 31st 00:00:00 in 2025 (non-leap year) should be DOY 365.0
        utc = "2025-12-31T00:00:00.000"
        et = spiceypy.str2et(utc)
        epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)

        doy = epoch_to_fractional_doy(epoch)

        assert doy == 365.0

    def test_fractional_time_components(self):
        """Test that hours, minutes, and seconds contribute correctly."""
        # January 2nd at 06:30:00 should be DOY 2.0 + 6.5/24 = 2.270833...
        utc = "2025-01-02T06:30:00.000"
        et = spiceypy.str2et(utc)
        epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)

        doy = epoch_to_fractional_doy(epoch)

        # Expected: 2.0 + 6/24 + 30/1440 = 2.0 + 0.25 + 0.02083... = 2.270833...
        expected_doy = 2.0 + 6.0 / 24.0 + 30.0 / 1440.0
        np.testing.assert_almost_equal(doy, expected_doy, decimal=4)

    def test_subsecond_precision(self):
        """Test that subsecond precision is preserved."""
        # January 1st at 00:00:00.5 should be DOY 1.0 + 0.5/86400
        utc = "2025-01-01T00:00:00.500"
        et = spiceypy.str2et(utc)
        epoch = int(spiceypy.unitim(et, "ET", "TT") * 1e9)

        doy = epoch_to_fractional_doy(epoch)

        # Expected: 1.0 + 0.5/86400 = 1.00000578703...
        expected_doy = 1.0 + 0.5 / 86400.0
        np.testing.assert_almost_equal(doy, expected_doy, decimal=4)

    def test_array_input(self):
        """Test that np.array as input works."""
        # Test various dates throughout the year
        test_dates = [
            "2025-01-01T00:00:00.000",
            "2025-06-15T12:30:45.123",
            "2025-12-31T23:59:59.999",
            "2024-02-29T12:00:00.000",  # Leap year
        ]

        ets = spiceypy.str2et(test_dates)
        epochs = et_to_ttj2000ns(ets)
        doys = epoch_to_fractional_doy(epochs)
        assert np.all(doys >= 1.0)
        assert np.all(doys < 367.0)


def test_yyyymmdd_to_ttj2000ns():
    """Verify YYYYMMDD dates convert to TTJ2000ns using UTC."""
    expected = np.int64(et_to_ttj2000ns(str_to_et("2026-01-01T00:00:00")))
    assert str_yyyymmdd_to_ttj2000ns("20260101") == expected


def test_yyyymmdd_to_ttj2000ns_invalid_date():
    """Verify a value error is raised when a date is invalid."""
    with pytest.raises(
        ValueError, match="Date 202601012 must be 8 characters long in yyyymmdd format."
    ):
        str_yyyymmdd_to_ttj2000ns("202601012")
