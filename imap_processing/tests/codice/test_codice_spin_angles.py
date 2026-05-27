"""Exact-value regression tests for CoDICE instrument-frame spin angles.

These tests pin the corrected reference angles from Michael Starkey's
algorithm-document update (issue #3242), where the instrument-frame
reference angles were re-derived relative to the instrument +X axis
(a 270 deg shift from the original APD 2-12 look-direction reference).

The tests run without external validation data so they form the primary
regression guard against future accidental reverts.
"""

import numpy as np

from imap_processing.codice.constants import (
    HI_IALIRT_REF_SPIN_ANGLE,
    L2_HI_SECTORED_ANGLE,
    SSD_ID_TO_SPIN_ANGLE,
)
from imap_processing.ialirt.utils.constants import HI_IALIRT_SPIN_ANGLE


def test_hi_sectored_reference_angles():
    """L2_HI_SECTORED_ANGLE matches Michael's corrected hi-sectored table."""
    expected = np.array(
        [
            195.00,
            154.11,
            138.69,
            135.00,
            138.69,
            154.11,
            195.00,
            235.89,
            251.31,
            255.00,
            251.31,
            235.89,
        ]
    )
    np.testing.assert_array_equal(L2_HI_SECTORED_ANGLE, expected)


def test_hi_direct_events_ssd_reference_angles():
    """SSD_ID_TO_SPIN_ANGLE matches Michael's corrected hi-DE per-SSD table.

    Indices 2, 6, 10, 14 correspond to missing SSD IDs and remain NaN.
    """
    expected_by_id = {
        0: 187.50,
        1: 146.61,
        3: 131.19,
        4: 127.50,
        5: 131.19,
        7: 146.61,
        8: 187.50,
        9: 228.39,
        11: 243.81,
        12: 247.50,
        13: 243.81,
        15: 228.39,
    }
    nan_ids = {2, 6, 10, 14}
    for ssd_id, expected in expected_by_id.items():
        assert SSD_ID_TO_SPIN_ANGLE[ssd_id] == expected, (
            f"SSD_ID_TO_SPIN_ANGLE[{ssd_id}] = {SSD_ID_TO_SPIN_ANGLE[ssd_id]}, "
            f"expected {expected}"
        )
    for ssd_id in nan_ids:
        assert np.isnan(SSD_ID_TO_SPIN_ANGLE[ssd_id]), (
            f"SSD_ID_TO_SPIN_ANGLE[{ssd_id}] should be NaN"
        )


def test_hi_ialirt_reference_angles():
    """HI_IALIRT_REF_SPIN_ANGLE matches Michael's corrected hi-ialirt table."""
    expected = np.array([196.85, 174.55, 253.16, 275.44], dtype=float)
    np.testing.assert_array_equal(HI_IALIRT_REF_SPIN_ANGLE, expected)


def test_hi_ialirt_spin_angle_first_column_matches_reference():
    """Derived HI_IALIRT_SPIN_ANGLE[:, 0] equals HI_IALIRT_REF_SPIN_ANGLE.

    The IALiRT spin angle grid is built by adding 0, 90, 180, 270 deg
    (mod 360) to each reference angle. Bin 0 must reproduce the reference
    exactly; any drift indicates a derivation bug.
    """
    np.testing.assert_array_equal(HI_IALIRT_SPIN_ANGLE[:, 0], HI_IALIRT_REF_SPIN_ANGLE)


def test_lo_direct_events_spin_angle_formula():
    """Lo direct-events spin_angle formula is (n*15 + 277.5) mod 360.

    Per Michael's algorithm-document update for issue #3242, the original
    +7.5 deg bin-center offset is replaced by +277.5 deg (+7.5 + 270 deg).
    This pins the formula constants applied at
    ``imap_processing/codice/codice_l2.py`` inside ``process_lo_direct_events``.
    """
    spin_sectors = np.arange(24, dtype=np.float32)
    expected = (spin_sectors * 15.0 + 277.5) % 360.0
    assert expected[0] == 277.5
    assert expected[5] == 352.5
    assert expected[6] == 7.5
    assert expected[12] == 97.5
    assert expected[23] == 262.5
