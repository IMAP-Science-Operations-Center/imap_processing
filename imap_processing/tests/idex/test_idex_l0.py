"""Tests the decommutation process for IDEX CCSDS Packets."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.idex.l1.idex_l1 import PacketParser


@pytest.fixture(scope="session")
def decom_test_data() -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xarray.Dataset
        A ``xarray`` dataset containing the test data
    """
    test_file = Path(
        f"{imap_module_directory}/tests/idex/imap_idex_l0_raw_20230725_v001.pkts"
    )
    return PacketParser(test_file, "001").data


def test_idex_decom_length(decom_test_data: xr.Dataset):
    """Verify that there are 6 data variables in the output.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    assert len(decom_test_data) == 42


def test_idex_decom_event_num(decom_test_data: xr.Dataset):
    """Verify that 19 impacts were gathered by the test data.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    for var in decom_test_data:
        assert len(decom_test_data[var]) == 19


def test_idex_tof_high_data(decom_test_data: xr.Dataset):
    """Verify that a sample of the data is correct.

    ``impact_14_tof_high_data.txt`` has been verified correct by the IDEX team

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """

    with open(f"{imap_module_directory}/tests/idex/impact_14_tof_high_data.txt") as f:
        data = np.array([int(line.rstrip("\n")) for line in f])
    assert (decom_test_data["TOF_High"][13].data == data).all()
