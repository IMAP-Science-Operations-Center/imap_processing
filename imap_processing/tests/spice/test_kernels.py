"""Tests coverage for imap_processing/spice/kernels.py"""

import pytest
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError

from imap_processing.spice import kernels


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
def test_ensure_spice_emus_mk_path(func, use_test_metakernel):
    """Test functionality of ensure spice with SPICE_METAKERNEL set"""
    assert func(577365941.184, "ISOC", 3) == "2018-04-18T23:24:31.998"


def test_ensure_spice_time_kernels():
    """Test functionality of ensure spice with timekernels set"""
    wrapped = kernels.ensure_spice(spice.et2utc, time_kernels_only=True)
    # TODO: Update/remove this test when a decision has been made about
    #   whether IMAP will use the time_kernels_only functionality and the
    #   ensure_spice decorator has been update.
    with pytest.raises(NotImplementedError):
        _ = wrapped(577365941.184, "ISOC", 3) == "2018-04-18T23:24:31.998"


def test_ensure_spice_key_error():
    """Test functionality of ensure spice when all branches fail"""
    wrapped = kernels.ensure_spice(spice.et2utc)
    # The ensure_spice decorator should raise a SpiceyError when all attempts to
    # furnish a set of kernels with sufficient coverage for the spiceypy
    # functions that it decorates.
    with pytest.raises(SpiceyError):
        _ = wrapped(577365941.184, "ISOC", 3) == "2018-04-18T23:24:31.998"
