"""Tests time functions for I-ALiRT instruments."""

import numpy as np

from imap_processing.ialirt.utils.time import calculate_time


def test_calculate_time():
    """Tests calculate_time function."""

    sc_sclk_sec = np.array([1, 2, 3, 4, 5])
    sc_sclk_sub_sec = np.array([0, 1, 2, 3, 4])

    time = calculate_time(sc_sclk_sec, sc_sclk_sub_sec, 256)

    np.testing.assert_allclose(
        time, sc_sclk_sec + sc_sclk_sub_sec / 256.0, atol=1e-03, rtol=0
    )
