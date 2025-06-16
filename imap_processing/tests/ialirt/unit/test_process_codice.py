"""Tests for the ``process_codice`` module.

See tests.codice.test_codice_l1a for more unit tests related to this code.
"""

from pathlib import Path

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.ialirt.l0.process_codice import process_codice

pytestmark = pytest.mark.external_test_data


@pytest.fixture(scope="session")
def l0_test_file():
    return Path(
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "imap_codice_l0_raw_20241110_v001.pkts"
    )


@pytest.fixture(scope="session")
def test_datasets(l0_test_file):
    return decom_packets(l0_test_file)


@pytest.fixture
def codice_test_data(apid, test_datasets):
    return test_datasets[apid]


@pytest.mark.parametrize(
    "apid", [1152, 1168]
)  # APIDs correspond to COD_LO_IAL and CO_HI_IAL
def test_process_codice(apid, codice_test_data, caplog):
    """Ensure that the ``process_codice`` function creates a dataset

    Here we just need to make sure the function is returning the expected data.
    CoDICE I-ALiRT data products are being validated separately in the
    ``codice.test_codice_l[1a|1b|2]`` modules.
    """

    with caplog.at_level("WARNING"):
        dataset = process_codice(codice_test_data)

    assert isinstance(dataset, xr.Dataset)
