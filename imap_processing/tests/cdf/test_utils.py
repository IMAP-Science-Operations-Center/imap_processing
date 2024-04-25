"""Tests for the ``cdf.utils`` module."""

from pathlib import Path

import imap_data_access
import numpy as np
import xarray as xr

from imap_processing import launch_time
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, load_cdf, write_cdf
from imap_processing.swe.swe_cdf_attrs import swe_l1a_global_attrs


def test_calc_start_time():
    """Tests the ``calc_start_time`` function"""

    assert calc_start_time(0) == launch_time
    assert calc_start_time(1) == launch_time + np.timedelta64(1, "s")


def test_load_cdf():
    """Tests the ``load_cdf`` function."""

    file_path = Path("data/imap_codice_l1a_hskp_20100101_v001.cdf")
    dataset = load_cdf(file_path)
    assert isinstance(dataset, xr.core.dataset.Dataset)


def test_write_cdf():
    """Tests the ``write_cdf`` function."""

    # Set up a fake dataset
    # lots of requirements on attributes, so depend on SWE for now
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
        attrs=swe_l1a_global_attrs.output()
        | {"Logical_source": "imap_swe_l1_sci", "Data_version": "001"},
    )
    dataset["epoch"].attrs = ConstantCoordinates.EPOCH

    file_path = write_cdf(dataset)
    assert file_path.exists()
    assert file_path.name == "imap_swe_l1_sci_20100101_v001.cdf"
    assert file_path.relative_to(imap_data_access.config["DATA_DIR"])
