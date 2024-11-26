"""Tests the decommutation process for IDEX CCSDS Packets."""

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory


def test_idex_decom_length(decom_test_data: xr.Dataset):
    """Verify that the output data has the expected number of data variables.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    assert len(decom_test_data) == 42


def test_idex_decom_event_num(decom_test_data: xr.Dataset):
    """Verify that 14 impacts were gathered by the test data.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    for var in decom_test_data:
        assert len(decom_test_data[var]) == 14


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
