"""Tests the L2 processing of CoDICE L1 data"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice.codice_l2 import (
    add_dataset_attributes,
    compute_geometric_factors,
)

pytestmark = pytest.mark.external_test_data

EXPECTED_LOGICAL_SOURCES = [
    "imap_codice_l2_hi-direct-events",
    "imap_codice_l2_lo-direct-events",
]


@pytest.fixture
def mock_cdf_attrs():
    # Create a mock ImapCdfAttributes object
    cdf_attrs = MagicMock(spec=ImapCdfAttributes)
    cdf_attrs.get_global_attributes.return_value = {
        "global_attr_key": "global_attr_value"
    }
    cdf_attrs.get_variable_attributes.side_effect = lambda var, check_schema: {
        "var1": {"attr1": "value1"},
        "test-product-var2": {"attr2": "value2"},
    }[var]
    return cdf_attrs


@pytest.fixture
def mock_half_spin_lut(monkeypatch):
    """
    Mock HALF_SPIN_LUT for testing.
    Example:
      ESA steps 0–63 belong to half_spin=1
      ESA steps 64–127 belong to half_spin=2
    """
    mock_lut = {
        1: list(range(0, 64)),
        2: list(range(64, 128)),
    }
    monkeypatch.setattr(
        "imap_processing.codice.codice_l2.HALF_SPIN_LUT",
        mock_lut,
    )


def test_compute_geometric_factors_all_full_mode(mock_half_spin_lut):
    # rgfo_half_spin = 3 means all half_spin values (1 or 2) are < rgfo_half_spin
    dataset = xr.Dataset({"rgfo_half_spin": (("epoch",), np.array([3, 3]))})

    result = compute_geometric_factors(dataset)

    # Expect 0.75 everywhere
    expected = np.full((2, 128), 0.75)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_all_reduced_mode(mock_half_spin_lut):
    # rgfo_half_spin = 0 means all half_spin values (>=1) are >= rgfo_half_spin
    dataset = xr.Dataset({"rgfo_half_spin": (("epoch",), np.array([0]))})

    result = compute_geometric_factors(dataset)

    # Expect 0.5 everywhere
    expected = np.full((1, 128), 0.5)
    np.testing.assert_array_equal(result, expected)


def test_compute_geometric_factors_mixed(mock_half_spin_lut):
    # rgfo_half_spin = 2
    dataset = xr.Dataset({"rgfo_half_spin": (("epoch",), np.array([2]))})

    result = compute_geometric_factors(dataset)

    # ESA steps 0-63 (half_spin=1) -> 1 < 2 → 0.75
    # ESA steps 64-127 (half_spin=2) -> 2 !< 2 → 0.5
    expected = np.array([[0.75] * 64 + [0.5] * 64])
    np.testing.assert_array_equal(result, expected)


def test_add_dataset_attributes(mock_cdf_attrs):
    dataset_name = "imap_codice_l2_test-product"

    # Create a sample xarray.Dataset
    sample_dataset = xr.Dataset(
        {
            "var1": (["dim1"], [1, 2, 3]),
            "var2": (["dim1"], [4, 5, 6]),
            "var3": (["dim1"], [7, 8, 9]),
        }
    )

    # Patch the logger to capture error messages
    with patch("imap_processing.codice.codice_l2.logger") as mock_logger:
        # Call the function
        updated_dataset = add_dataset_attributes(
            sample_dataset, dataset_name, mock_cdf_attrs
        )

        # Assert global attributes are updated
        assert updated_dataset.attrs == {"global_attr_key": "global_attr_value"}

        # Assert variable attributes are updated

        # var1 should get attributes directly
        assert updated_dataset["var1"].attrs == {"attr1": "value1"}

        # var2 should get attributes with product descriptor prefix (test-product)
        assert updated_dataset["var2"].attrs == {"attr2": "value2"}

        # var3 should log an error since it doesn't have corresponding attributes
        assert updated_dataset["var3"].attrs == {}

        # Check logger error call for missing attributes
        mock_logger.error.assert_called_with(
            "Field 'var3' and 'test-product-var3' not found in attribute manager."
        )
