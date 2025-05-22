"""Tests for the ``process_codicelo`` module.

See tests.codice.test_codice_l1a for more unit tests related to this code.
"""

from pathlib import Path

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.ialirt.l0.process_codicelo import process_codicelo

pytestmark = pytest.mark.external_test_data


@pytest.fixture(scope="session")
def codicelo_test_data():
    """Returns the CoDICE-lo I-ALiRT test dataset"""
    l0_test_file = Path(
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "imap_codice_l0_raw_20241110_v001.pkts"
    )
    datasets = decom_packets(l0_test_file)
    codicelo_test_data = datasets[1152]

    return codicelo_test_data


@pytest.fixture(scope="session")
def codicelo_validation_data():
    """Returns the validation dataset."""

    data_path = Path(
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "validation"
        / "imap_codice_l1a_lo-ialirt_20241110193700_v0.0.0.cdf"
    )
    data = load_cdf(data_path)

    return data


def test_process_codicelo(codicelo_test_data, codicelo_validation_data, caplog):
    """Tests ``process_codicelo``."""

    with caplog.at_level("WARNING"):
        dataset = process_codicelo(codicelo_test_data)
    assert isinstance(dataset, xr.core.dataset.Dataset)
