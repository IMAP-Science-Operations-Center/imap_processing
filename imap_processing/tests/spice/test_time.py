"""Tests coverage for imap_processing/spice/time.py"""

import numpy as np
import pytest
import spiceypy as spice

from imap_processing.spice import IMAP_SC_ID
from imap_processing.spice.time import TICK_DURATION, _sct2e_wrapper, met_to_j2000ns


def test_met_to_j2000ns(furnish_sclk, furnish_test_lsk):
    """Test coverage for met_to_j2000ns function."""
    utc = "2026-01-01T00:00:00.125"
    et = spice.str2et(utc)
    sclk_str = spice.sce2s(IMAP_SC_ID, et)
    seconds, ticks = sclk_str.split("/")[1].split(":")
    met = float(seconds) + float(ticks) * TICK_DURATION
    j2000ns = met_to_j2000ns(met)
    assert j2000ns == et * 1e9


@pytest.mark.parametrize("sclk_ticks", [0.0, np.arange(10)])
def test_sct2e_wrapper(sclk_ticks, furnish_sclk, furnish_test_lsk):
    """Test for `_sct2e_wrapper` function."""
    et = _sct2e_wrapper(sclk_ticks)
    if isinstance(sclk_ticks, float):
        assert isinstance(et, float)
    else:
        assert len(et) == len(sclk_ticks)
