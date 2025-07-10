"""Tests for the ``process_codice`` module.

See tests.codice.test_codice_l[1a|1b|2] for more unit tests related to this
code.
"""

from pathlib import Path

import pytest

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_codice import process_codice
from imap_processing.utils import packet_file_to_datasets

pytestmark = pytest.mark.external_test_data


@pytest.fixture(scope="session")
def l0_test_file():
    return Path(
        imap_module_directory / "tests" / "ialirt" / "data" / "l0" / "apid_478.bin"
    )


@pytest.fixture(scope="session")
def test_datasets(l0_test_file):
    xtce_packet_definition = Path(
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"
    )

    datasets = packet_file_to_datasets(l0_test_file, xtce_packet_definition)

    return datasets


@pytest.fixture
def codice_test_data(test_datasets):
    return test_datasets[478]


def test_process_codice(codice_test_data, caplog):
    """Ensure that the ``process_codice`` function creates a dataset

    Here we just need to make sure the function is returning the expected data.
    CoDICE I-ALiRT data products are being validated separately in the
    ``codice.test_codice_l[1a|1b|2]`` modules.
    """

    with caplog.at_level("WARNING"):
        cod_lo_data, cod_hi_data = process_codice(codice_test_data)

    assert isinstance(cod_lo_data, list)
    assert all(isinstance(item, dict) for item in cod_lo_data)
    assert isinstance(cod_hi_data, list)
    assert all(isinstance(item, dict) for item in cod_hi_data)
