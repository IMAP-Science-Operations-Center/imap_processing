"""Tests for the ``cdf.utils`` module."""

import imap_data_access
import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf import epoch_attrs
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import (
    IMAP_EPOCH,
    J2000_EPOCH,
    load_cdf,
    met_to_j2000ns,
    write_cdf,
)


@pytest.fixture()
def test_dataset():
    """Create a simple ``xarray`` dataset to be used in testing

    Returns
    -------
    dataset : xarray.Dataset
        The ``xarray`` dataset object
    """
    # Load the CDF attrs
    swe_attrs = ImapCdfAttributes()
    swe_attrs.add_instrument_global_attrs("swe")
    swe_attrs.add_global_attribute("Data_version", "001")

    dataset = xr.Dataset(
        {
            "epoch": (
                "epoch",
                met_to_j2000ns([1, 2, 3]),
            )
        },
        attrs=swe_attrs.get_global_attributes("imap_swe_l1a_sci")
        | {
            "Logical_file_id": "imap_swe_l1a_sci_20100101_v001",
        },
    )
    dataset["epoch"].attrs = epoch_attrs
    dataset["epoch"].attrs["DEPEND_0"] = "epoch"

    return dataset


def test_met_to_j2000ns():
    """Tests the ``met_to_j2000ns`` function"""
    imap_epoch_offset = (IMAP_EPOCH - J2000_EPOCH).astype(np.int64)
    assert met_to_j2000ns(0) == imap_epoch_offset
    assert met_to_j2000ns(1) == imap_epoch_offset + 1e9
    # Large input should work (avoid overflow with int32 SHCOARSE inputs)
    assert met_to_j2000ns(np.int32(2**30)) == imap_epoch_offset + 2**30 * 1e9
    assert met_to_j2000ns(0).dtype == np.int64
    # Float input should work
    assert met_to_j2000ns(0.0) == imap_epoch_offset
    assert met_to_j2000ns(1.2) == imap_epoch_offset + 1.2e9
    # Negative input should work
    assert met_to_j2000ns(-1) == imap_epoch_offset - 1e9
    # array-like input should work
    output = met_to_j2000ns([0, 1])
    np.testing.assert_array_equal(output, [imap_epoch_offset, imap_epoch_offset + 1e9])
    # Different reference epoch should shift the result
    different_epoch_time = IMAP_EPOCH + np.timedelta64(2, "ns")
    assert (
        met_to_j2000ns(0, reference_epoch=different_epoch_time) == imap_epoch_offset + 2
    )


def test_load_cdf(test_dataset):
    """Tests the ``load_cdf`` function."""

    # Write the dataset to a CDF to be used to test the load function
    file_path = write_cdf(test_dataset)

    # Load the CDF and ensure the function returns a dataset
    dataset = load_cdf(file_path)
    assert isinstance(dataset, xr.core.dataset.Dataset)

    # Test that epoch is represented as a 64bit integer
    assert dataset["epoch"].data.dtype == np.int64
    # Test removal of attributes that are added on by cdf_to_xarray and
    # are specific to xarray plotting
    xarray_attrs = ["units", "standard_name", "long_name"]
    for _, data_array in dataset.variables.items():
        for attr in xarray_attrs:
            assert attr not in data_array.attrs


def test_write_cdf(test_dataset):
    """Tests the ``write_cdf`` function.

    Parameters
    ----------
    dataset : xarray.Dataset
        An ``xarray`` dataset object to test with
    """

    file_path = write_cdf(test_dataset)
    assert file_path.exists()
    assert file_path.name == "imap_swe_l1a_sci_20100101_v001.cdf"
    assert file_path.relative_to(imap_data_access.config["DATA_DIR"])


def test_written_and_loaded_dataset(test_dataset):
    """Tests that a dataset that is written to CDF and then loaded results in
    the original dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        An ``xarray`` dataset object to test with
    """

    new_dataset = load_cdf(write_cdf(test_dataset))
    assert str(test_dataset) == str(new_dataset)
