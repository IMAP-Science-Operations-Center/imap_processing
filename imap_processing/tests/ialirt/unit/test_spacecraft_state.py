"""Module to test attitude calculations."""

import numpy as np
import pytest
import spiceypy

from imap_processing.ialirt.utils.spacecraft_state import calculate_gsm_state
from imap_processing.spice.geometry import SpiceBody, SpiceFrame
from imap_processing.spice.time import met_to_sclkticks, sct_to_et


@pytest.mark.external_kernel
def test_calculate_gsm_state(furnish_kernels):
    """Tests calculate_gsm_state function."""
    spiceypy.kclear()

    kernels = [
        # Planetary ephemeris kernel
        "de440s.bsp",
        # Frames kernel containing IMAP_GSM frame
        "imap_science_v100.tf",
        # Planetary constants kernel
        "pck00011.tpc",
        # TODO: Oddly the code runs w/out the below kernel.
        #  Figure out why.
        # IMAP spacecraft SPK kernel
        "imap_spk_demo.bsp",
    ]
    with furnish_kernels(kernels):
        met = np.array([482371202], dtype=np.int64)
        sclk_ticks = met_to_sclkticks(met)
        et = sct_to_et(sclk_ticks)
        gsm_position, gsm_velocity = calculate_gsm_state(met)
        state, _ = spiceypy.spkezr(
            SpiceBody.IMAP.name,
            et,
            SpiceFrame.IMAP_GSM.name,
            "None",
            SpiceBody.SUN.name,
        )
    assert np.allclose(gsm_position, state[0][:3], atol=1e-6)
    assert np.allclose(gsm_velocity, state[0][3:], atol=1e-6)
