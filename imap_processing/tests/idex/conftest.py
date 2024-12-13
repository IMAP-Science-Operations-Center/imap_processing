from pathlib import Path

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.idex.idex_l1a import PacketParser


@pytest.fixture()
def decom_test_data() -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    dataset : xarray.Dataset
        A ``xarray`` dataset containing the test data
    """
    test_file = Path(
        f"{imap_module_directory}/tests/idex/imap_idex_l0_raw_20231214_v001.pkts"
    )
    return PacketParser(test_file, "001").data
