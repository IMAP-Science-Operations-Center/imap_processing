"""Tests for the ``cdf.utils`` module."""

from pathlib import Path

import imap_data_access
import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import (
    load_cdf,
    write_cdf,
)
from imap_processing.spice.time import met_to_j2000ns


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
    dataset["epoch"].attrs = swe_attrs.get_variable_attributes("epoch")
    dataset["epoch"].attrs["DEPEND_0"] = "epoch"

    return dataset


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


def test_parents_injection(test_dataset):
    """Tests the ``write_cdf`` function for Parents attribute injection.

    Parameters
    ----------
    test_dataset : xarray.Dataset
        An ``xarray`` dataset object to test with
    """
    parent_paths = [Path("test_parent1.cdf"), Path("/abc/test_parent2.cdf")]
    new_dataset = load_cdf(write_cdf(test_dataset, parent_files=parent_paths))
    assert new_dataset.attrs["Parents"] == [p.name for p in parent_paths]
