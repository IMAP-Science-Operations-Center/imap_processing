"""Tests for the ``cdf.utils`` module."""

from unittest import mock

import imap_data_access
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import (
    load_cdf,
    parse_filename_like,
    write_cdf,
)
from imap_processing.spice.time import met_to_ttj2000ns


@pytest.fixture
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
                met_to_ttj2000ns([1, 2, 3]),
            ),
            "nan_data": ("epoch", np.array([1.0, 2.0, np.nan]), {"FILLVAL": -1.0e31}),
        },
        attrs=swe_attrs.get_global_attributes("imap_swe_l1a_sci")
        | {
            "Logical_file_id": "imap_swe_l1a_sci_20100101_v001",
        },
    )
    dataset["epoch"].attrs = swe_attrs.get_variable_attributes(
        "epoch", check_schema=False
    )

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

    assert np.isnan(dataset["nan_data"].data[2])


def test_load_cdf_extra_kwargs(test_dataset):
    """Test that load_cdf passes the correct extra kwargs to xarray_to_cdf"""
    # Write the dataset to a CDF to be used to test the load function
    file_path = write_cdf(test_dataset)
    with mock.patch(
        "imap_processing.cdf.utils.cdf_to_xarray", autospec=True
    ) as mock_cdf_to_xarray:
        load_cdf(file_path, to_datetime=False)
        assert mock_cdf_to_xarray.call_args.kwargs["to_datetime"] is False


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


def test_repoint_start_date(test_dataset):
    output_file_path = write_cdf(test_dataset)
    assert "imap_swe_l1a_sci_20100101_v001.cdf" in output_file_path.name
    test_dataset.attrs["Start_date"] = "20001212"

    output_file_path = write_cdf(test_dataset)
    assert "imap_swe_l1a_sci_20001212_v001.cdf" in output_file_path.name

    test_dataset.attrs["Repointing"] = "12345"
    output_file_path = write_cdf(test_dataset)
    assert "imap_swe_l1a_sci_20001212-repoint12345_v001.cdf" in output_file_path.name


def test_write_cdf_extra_cdf_kwargs(test_dataset):
    """Test the kwargs passed to cdflib.xarray.xarray_to_cdf by write_cdf()"""
    with mock.patch(
        "imap_processing.cdf.utils.xarray_to_cdf", autospec=True
    ) as xarray_to_cdf:
        write_cdf(test_dataset)
        assert xarray_to_cdf.call_args.kwargs["terminate_on_warning"] is False
        assert xarray_to_cdf.call_args.kwargs["compression"] == 6
        test_dataset.attrs["Logical_source"] = "imap_swe_l2_sci"
        write_cdf(test_dataset, compression=9)
        assert xarray_to_cdf.call_args.kwargs["terminate_on_warning"] is True
        assert xarray_to_cdf.call_args.kwargs["istp"] is True
        assert xarray_to_cdf.call_args.kwargs["compression"] == 9


def test_write_cdf_converts_extension_array(test_dataset):
    """Convert extension arrays for cdflib without changing the input dataset."""
    test_dataset["labels"] = (
        "label",
        pd.array(["first", "second"], dtype="string"),
    )

    with mock.patch(
        "imap_processing.cdf.utils.xarray_to_cdf", autospec=True
    ) as xarray_to_cdf:
        write_cdf(test_dataset)

    converted_dataset = xarray_to_cdf.call_args.args[0]
    assert isinstance(converted_dataset["labels"].data, np.ndarray)
    np.testing.assert_array_equal(converted_dataset["labels"], ["first", "second"])
    assert isinstance(test_dataset["labels"].data, pd.api.extensions.ExtensionArray)


@pytest.mark.parametrize(
    "test_str, compare_dict",
    [
        (
            "imap_hi_l1b_45sensor-de",
            {
                "mission": "imap",
                "instrument": "hi",
                "data_level": "l1b",
                "sensor": "45sensor",
                "descriptor": "de",
            },
        ),
        (
            "imap_hi_l1a_hist_20250415_v001",
            {
                "mission": "imap",
                "instrument": "hi",
                "data_level": "l1a",
                "descriptor": "hist",
                "start_date": "20250415",
                "version": "v001",
            },
        ),
        (
            "imap_hi_l1c_90sensor-pset_20250415-repoint12345_v001.cdf",
            {
                "mission": "imap",
                "instrument": "hi",
                "data_level": "l1c",
                "sensor": "90sensor",
                "descriptor": "pset",
                "start_date": "20250415",
                "repointing": "12345",
                "version": "v001",
                "extension": "cdf",
            },
        ),
        ("foo_hi_l1c_90sensor-pset_20250415_v001.cdf", None),
        ("imap_hi_l1c", None),
    ],
)
def test_parse_filename_like(test_str, compare_dict):
    """Test coverage for parse_filename_like function"""
    if compare_dict:
        match = parse_filename_like(test_str)
        for key, value in compare_dict.items():
            assert match[key] == value
    else:
        with pytest.raises(ValueError, match="Filename like string did not contain"):
            _ = parse_filename_like(test_str)
