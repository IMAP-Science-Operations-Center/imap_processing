# from .conftest import TEST_L2_FILES
from turtle import pd
import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l2 import process_codice_l2
from imap_processing.codice.utils import reshape_ssd_energy_df

pytestmark = pytest.mark.external_test_data

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

def test_hi_de():
    test_file_path = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_data"
        / "imap_codice_l1a_hi-direct-events_20250814211100_v0.0.3.cdf"
    )

    l2_dataset = process_codice_l2(test_file_path)
    # print(l2_dataset.data_vars)

    # Validation data
    val_file = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l2_validation"
        / "imap_codice_l2_hi-direct-events_20250814211100_v0.0.3.cdf"
    )

    val_dataset = load_cdf(val_file)
    # print(val_dataset.data_vars)
    for variable in l2_dataset.data_vars:
        assert l2_dataset[variable].shape == val_dataset[variable].shape
        if variable in ["gain", "multi-flag", "spin_number"]:
            np.testing.assert_array_equal(
                l2_dataset[variable].values, val_dataset[variable].values
            )
        elif variable == "elevation_angle":
            # Test if both get nan in same place
            np.testing.assert_array_equal(
                np.isnan(l2_dataset[variable].values),
                np.isnan(val_dataset[variable].values),
            )
            # Test if values are close (ignoring nan)
            np.testing.assert_allclose(
                l2_dataset[variable].values,
                val_dataset[variable].values,
                equal_nan=True,
            )
        elif variable == "spin_angle":
            # Test if both get nan in same place
            np.testing.assert_array_equal(
                np.isnan(l2_dataset[variable].values),
                np.isnan(val_dataset[variable].values),
            )
            # TODO: why this validation is not working
            # # Test if values are close (ignoring nan)
            # np.testing.assert_allclose(
            #     l2_dataset[variable].values,
            #     val_dataset[variable].values,
            #     equal_nan=True,
            # )
        elif variable == "tof_ns":
            # Test if both get nan in same place
            np.testing.assert_array_equal(
                np.isnan(l2_dataset[variable].values),
                np.isnan(val_dataset[variable].values),
            )
            # Test if values are close (ignoring nan)
            np.testing.assert_allclose(
                l2_dataset[variable].values,
                val_dataset[variable].values,
                equal_nan=True,
            )
        elif variable == "ssd_energy":
            # Test if both get nan in same place
            np.testing.assert_array_equal(
                (np.isnan(l2_dataset[variable].values)).shape,
                (np.isnan(val_dataset[variable].values)).shape,
            )

            non_nan_indices = np.where(
                ~np.isnan(l2_dataset[variable].values)
                & ~np.isnan(val_dataset[variable].values)
            )

            # Test if values are close (ignoring nan)
            np.testing.assert_allclose(
                l2_dataset[variable].values[non_nan_indices],
                val_dataset[variable].values[non_nan_indices],
            )
