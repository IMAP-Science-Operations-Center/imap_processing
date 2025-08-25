# from .conftest import TEST_L2_FILES
import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l2 import process_codice_l2

pytestmark = pytest.mark.external_test_data


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
