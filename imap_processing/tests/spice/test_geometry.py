"""Tests coverage for imap_processing/spice/geometry.py"""

import numpy as np

from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
    ensured_spkezr,
    imap_state,
)


def test_imap_state(use_test_metakernel):
    """Test coverage for imap_state()"""
    et = np.linspace(798033670, 798034670)
    state = imap_state(et, observer=SpiceBody.EARTH)
    assert len(state) == len(et)


def test_ensured_spkezr(use_test_metakernel):
    """Test coverage for ensured_spkezr()"""
    # The imap_spk_demo.bsp kernel provides ephemeris coverage for 2025-04-15 to
    # 2026-04-16. The kernel provides the IMAP ephemeris relative to Earth, so
    # only the position relative to Earth can be queried without loading
    # additional kernels.
    # The queried ET, 798033670 is ~2025-04-16T00:00:00.0
    state, lt = ensured_spkezr(
        SpiceBody.IMAP.name,
        798033670,
        SpiceFrame.ECLIPJ2000.name,
        "NONE",
        SpiceBody.EARTH.name,
    )
    assert state.shape == (6,)
    assert isinstance(lt, float)
