import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.ialirt.l0.process_codicehi import (
    append_cod_hi_data,
    find_groups,
    process_codicehi,
)
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_codicehi_path():
    """Returns the xtce directory."""
    return (
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_codicehi.xml"
    )


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the xtce directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "hi_fsw_view_1_ccsds.bin"
    )


@pytest.fixture(scope="session")
def codicehi_validation_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "imap_codice_l1a_hi-ialirt_20240523200000_v0.0.0.cdf"
    )
    data = load_cdf(data_path)

    return data


@pytest.fixture()
def codicehi_test_data(binary_packet_path, xtce_codicehi_path):
    """Create xarray data"""
    apid = 1168
    codicehi_test_data = packet_file_to_datasets(
        binary_packet_path, xtce_codicehi_path
    )[apid]

    return codicehi_test_data


def test_find_groups(codicehi_test_data):
    """Tests find_groups"""

    grouped_data = find_groups(codicehi_test_data)
    unique_groups = np.unique(grouped_data["group"])
    for group in unique_groups:
        group_data = grouped_data["cod_hi_counter"].values[
            grouped_data["group"] == group
        ]
        np.testing.assert_array_equal(group_data, np.arange(234))


def test_append_cod_hi_data(codicehi_test_data):
    """Tests append_cod_hi_data"""

    grouped_data = find_groups(codicehi_test_data)
    unique_groups = np.unique(grouped_data["group"])
    for group in unique_groups:
        mask = grouped_data["group"] == group
        filtered_indices = np.where(mask)[0]
        group_data = grouped_data.isel(epoch=filtered_indices)
        expected_cod_hi_counter = np.repeat(group_data["cod_hi_counter"].values, 5)
        appended_data = append_cod_hi_data(group_data)
        assert np.array_equal(
            appended_data["cod_hi_counter"].values, expected_cod_hi_counter.astype(int)
        )


def test_process_codicehi(codicehi_test_data, codicehi_validation_data, caplog):
    """Tests process_codicehi."""
    codicehi_product = process_codicehi(codicehi_test_data)
    assert codicehi_product == [{}]

    indices = (codicehi_test_data["cod_hi_acq"] != 0).values.nonzero()[0]
    codicehi_test_data["cod_hi_counter"].values[indices[0] : indices[0] + 234] = (
        np.random.permutation(234)
    )

    with caplog.at_level("WARNING"):
        process_codicehi(codicehi_test_data)

    assert any(
        "does not contain all values from 0 to 233 without duplicates" in message
        for message in caplog.text.splitlines()
    )
