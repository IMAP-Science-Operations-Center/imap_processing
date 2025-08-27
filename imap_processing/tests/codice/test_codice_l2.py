"""Tests the L2 processing of CoDICE L1 data"""

from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice.codice_l2 import add_dataset_attributes, process_codice_l2

from .conftest import TEST_L2_FILES

pytestmark = pytest.mark.external_test_data

EXPECTED_LOGICAL_SOURCES = [
    "imap_codice_l2_hi-direct-events",
    "imap_codice_l2_lo-direct-events",
]


@pytest.fixture(params=TEST_L2_FILES)
def test_l2_data(request) -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xr.Dataset
        A ``xarray`` dataset containing the test data
    """
    dataset = process_codice_l2(request.param)
    return dataset


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


@pytest.mark.parametrize(
    "test_l2_data, expected_logical_source",
    list(zip(TEST_L2_FILES, EXPECTED_LOGICAL_SOURCES, strict=False)),
    indirect=["test_l2_data"],
)
def test_l2_logical_sources(test_l2_data: xr.Dataset, expected_logical_source: str):
    """Tests that the ``process_codice_l2`` function generates datasets
    with the expected logical source.

    Parameters
    ----------
    test_l2_data : xr.Dataset
        A ``xarray`` dataset containing the test data
    expected_logical_source : str
        The expected CDF filename
    """

    dataset = test_l2_data

    assert dataset.attrs["Logical_source"] == expected_logical_source


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
