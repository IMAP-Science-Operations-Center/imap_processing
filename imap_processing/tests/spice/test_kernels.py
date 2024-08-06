"""Tests coverage for imap_processing/spice/kernels.py"""

import pytest
import spiceypy as spice

from imap_processing import kernels


@kernels.ensure_spice
def single_wrap_et2utc(et, fmt, prec):
    """Directly decorate a spice function with ensure_spice for use in tests"""
    return spice.et2utc(et, fmt, prec)


@kernels.ensure_spice
def double_wrap_et2utc(et, fmt, prec):
    """Decorate a spice function twice with ensure_spice for use in tests. This
    simulates some decorated outer functions that call lower level functions
    that are already decorated."""
    return single_wrap_et2utc(et, fmt, prec)


@kernels.ensure_spice(time_kernels_only=True)
def single_wrap_et2utc_tk_only(et, fmt, prec):
    """Directly wrap a spice function with optional time_kernels_only set True"""
    return spice.et2utc(et, fmt, prec)


@kernels.ensure_spice(time_kernels_only=True)
def double_wrap_et2utc_tk_only(et, fmt, prec):
    """Decorate a spice function twice with ensure_spice for use in tests. This
    simulates some decorated outer functions that call lower level functions
    that are already decorated."""
    return single_wrap_et2utc(et, fmt, prec)


@pytest.mark.parametrize(
    "func",
    [
        single_wrap_et2utc,
        single_wrap_et2utc_tk_only,
        double_wrap_et2utc,
        double_wrap_et2utc_tk_only,
    ],
)
def test_ensure_spice_emus_mk_path(func):
    """Test functionality of ensure spice with EMUS_METAKERNEL_PATH set"""
    assert func(577365941.184, "ISOC", 3) == "2018-04-18T23:24:31.998"
