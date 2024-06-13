"""Tests the L1 processing for decommutated IDEX data"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cdflib.xarray.xarray_to_cdf import ISTPError

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.idex.idex_packet_parser import PacketParser


@pytest.fixture()
def decom_test_data():
    test_file = Path(
        f"{imap_module_directory}/tests/idex/imap_idex_l0_raw_20230725_v001.pkts"
    )
    return PacketParser(test_file, "v001").data


def test_idex_cdf_file(decom_test_data):
    # Verify that a CDF file can be created with no errors thrown by xarray_to_cdf

    file_name = write_cdf(decom_test_data)

    assert file_name.exists()
    assert file_name.name == "imap_idex_l1_sci_20250724_v001.cdf"


def test_bad_cdf_attributes(decom_test_data):
    # Deliberately mess up the attributes to verify that an ISTPError is raised
    del decom_test_data["TOF_High"].attrs["DEPEND_1"]

    with pytest.raises(ISTPError):
        write_cdf(decom_test_data)


def test_bad_cdf_file_data(decom_test_data):
    # Deliberately mess up the data to verify that an ISTPError is raised
    bad_data_attrs = {
        "CATDESC": "Bad_Data",
        "DEPEND_0": "epoch",
        "DISPLAY_TYPE": "no_plot",
        "FIELDNAM": "Bad_Data",
        "FILLVAL": "",
        "FORMAT": "E12.2",
        "LABLAXIS": "Bad_Data",
        "UNITS": "",
        "VALIDMIN": "1",
        "VALIDMAX": "50",
        "VAR_TYPE": "support_data",
        "VAR_NOTES": """How did this data end up in here?
                        The CDF creation better fail.""",
    }
    bad_data_xr = xr.DataArray(
        name="bad_data",
        data=np.linspace(1, 50, 50),
        dims=("bad_data"),
        attrs=bad_data_attrs,
    )
    decom_test_data["Bad_data"] = bad_data_xr

    with pytest.raises(ISTPError):
        write_cdf(decom_test_data)


def test_idex_tof_high_data_from_cdf(decom_test_data):
    # Verify that a sample of the data is correct inside the CDF file
    # impact_14_tof_high_data.txt has been verified correct by the IDEX team
    with open(f"{imap_module_directory}/tests/idex/impact_14_tof_high_data.txt") as f:
        data = np.array([int(line.rstrip()) for line in f])

    file_name = write_cdf(decom_test_data)
    l1_data = load_cdf(file_name)
    assert (l1_data["TOF_High"][13].data == data).all()
