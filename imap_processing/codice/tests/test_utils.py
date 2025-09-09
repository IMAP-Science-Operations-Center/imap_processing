"""Tests for the codice utils module."""
import numpy as np
import pandas as pd
import pytest

from imap_processing.codice.utils import reshape_ssd_energy_df


def test_reshape_ssd_energy_df():
    """Test the reshape_ssd_energy_df function."""
    # Create a simple test dataframe
    test_df = pd.DataFrame({
        "bin_num": [0, 1, 2],
        "SSD 0 - LG": [10.0, 20.0, 30.0],
        "SSD 0 - MG": [11.0, 21.0, 31.0],
        "SSD 0 - HG": [12.0, 22.0, 32.0],
        "SSD 1 - LG": [13.0, 23.0, 33.0],
        "SSD 1 - MG": [14.0, 24.0, 34.0],
        "SSD 1 - HG": [15.0, 25.0, 35.0],
    })

    # Reshape the dataframe
    result = reshape_ssd_energy_df(test_df)

    # Check the shape of the result
    assert result.shape == (3, 16, 3)

    # Check the values for the first two SSDs
    np.testing.assert_almost_equal(result[0, 0, 0], 10.0)  # SSD 0, LG, bin 0
    np.testing.assert_almost_equal(result[1, 0, 0], 20.0)  # SSD 0, LG, bin 1
    np.testing.assert_almost_equal(result[2, 0, 0], 30.0)  # SSD 0, LG, bin 2
    
    np.testing.assert_almost_equal(result[0, 0, 1], 11.0)  # SSD 0, MG, bin 0
    np.testing.assert_almost_equal(result[0, 0, 2], 12.0)  # SSD 0, HG, bin 0
    
    np.testing.assert_almost_equal(result[0, 1, 0], 13.0)  # SSD 1, LG, bin 0
    np.testing.assert_almost_equal(result[0, 1, 1], 14.0)  # SSD 1, MG, bin 0
    np.testing.assert_almost_equal(result[0, 1, 2], 15.0)  # SSD 1, HG, bin 0

    # Check that missing SSDs are filled with NaN
    assert np.isnan(result[0, 2, 0])  # SSD 2, LG, bin 0
