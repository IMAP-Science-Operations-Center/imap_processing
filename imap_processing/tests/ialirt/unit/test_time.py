"""Tests grouping functions for I-ALiRT instruments."""

import numpy as np

from imap_processing.ialirt.utils.time import calculate_time


def test_calculate_time():
    """Tests calculate_time function."""

    coarse = np.array([1, 2, 3, 4, 5])
    fine = np.array([0, 1, 2, 3, 4])

    time = calculate_time(coarse, fine, 6553)

    np.testing.assert_allclose(time, coarse + fine / 65535.0, atol=1e-03, rtol=0)
