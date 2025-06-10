"""Tests the decommutation process for IDEX CCSDS Packets."""

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory


def test_idex_decom_length(decom_test_data_sci: xr.Dataset):
    """Verify that the output data has the expected number of data variables.

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        The dataset to test with
    """
    assert len(decom_test_data_sci) == 110


def test_idex_decom_event_num(decom_test_data_sci: xr.Dataset):
    """Verify that 14 impacts were gathered by the test data.

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        The dataset to test with
    """
    for var in decom_test_data_sci:
        if "epoch" in decom_test_data_sci[var].dims:
            assert len(decom_test_data_sci[var]) == 14


def test_idex_tof_high_data(decom_test_data_sci: xr.Dataset):
    """Verify that a sample of the data is correct.

    ``impact_14_tof_high_data.txt`` has been verified correct by the IDEX team

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        The dataset to test with
    """

    with open(
        f"{imap_module_directory}/tests/idex/test_data/impact_14_tof_high_data.txt"
    ) as f:
        data = np.array([int(line.rstrip("\n")) for line in f])
    assert (decom_test_data_sci["TOF_High"][13].data == data).all()


def test_catlst_event_num(decom_test_data_catlst: list[xr.Dataset]):
    """Verify that a sample of the data is correct.

    Parameters
    ----------
    decom_test_data_catlst : list[xarray.Dataset]
        The raw and derived (l1a and l1b) datasets to test with.
    """
    # test both l1a and l1b datasets
    for ds in decom_test_data_catlst:
        assert len(ds["epoch"]) == 1


def test_evt_event_num(decom_test_data_evt: list[xr.Dataset]):
    """Verify that a sample of the data is correct.

    Parameters
    ----------
    decom_test_data_evt : list[xarray.Dataset]
        The raw and derived (l1a and l1b) datasets to test with.
    """
    for ds in decom_test_data_evt:
        assert len(ds["epoch"]) == 28
