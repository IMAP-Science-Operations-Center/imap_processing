"""Tests for the ``cdf.utils`` module."""

import imap_data_access
import numpy as np
import pytest
import xarray as xr

from imap_processing import launch_time
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import calc_start_time, load_cdf, write_cdf


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

    dataset = xr.Dataset(
        {
            "epoch": (
                "epoch",
                [
                    np.datetime64("2010-01-01T00:01:01", "ns"),
                    np.datetime64("2010-01-01T00:01:02", "ns"),
                    np.datetime64("2010-01-01T00:01:03", "ns"),
                ],
            )
        },
        attrs=swe_attrs.get_global_attributes("imap_swe_l1a_sci")
        | {
            "Logical_file_id": "imap_swe_l1a_sci_20100101_v001",
        },
    )
    dataset["epoch"].attrs = ConstantCoordinates.EPOCH
    dataset["epoch"].attrs["DEPEND_0"] = "epoch"

    return dataset


def test_calc_start_time():
    """Tests the ``calc_start_time`` function"""

    assert calc_start_time(0) == launch_time
    assert calc_start_time(1) == launch_time + np.timedelta64(1, "s")
    different_launch_time = launch_time + np.timedelta64(2, "s")
    assert calc_start_time(
        0, launch_time=different_launch_time
    ) == launch_time + np.timedelta64(2, "s")


def test_load_cdf(test_dataset):
    """Tests the ``load_cdf`` function."""

    # Write the dataset to a CDF to be used to test the load function
    file_path = write_cdf(test_dataset)

    # Load the CDF and ensure the function returns a dataset
    dataset = load_cdf(file_path)
    assert isinstance(dataset, xr.core.dataset.Dataset)


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

    new_dataset = load_cdf(write_cdf(test_dataset), to_datetime=True)
    assert str(test_dataset) == str(new_dataset)
